# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-tune LLM
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Retrieve task parameters

# COMMAND ----------

input_model = dbutils.jobs.taskValues.get(taskKey = "Configure_env", key = 'input_model =', default = 'meta-llama/Llama-2-7b-chat-hf')
config_file_name = dbutils.jobs.taskValues.get(taskKey = "Configure_env", key = 'config_file_name', default = '/Workspace/Users/brett.smerdon@energyq.com.au/EQL-LLM/Jeff/configs/a10_config.json')

local_output_dir = dbutils.jobs.taskValues.get(taskKey = "Configure_env", key = 'local_output_dir')
dbfs_output_dir = dbutils.jobs.taskValues.get(taskKey = "Configure_env", key = 'dbfs_output_dir')
batch_size = dbutils.jobs.taskValues.get(taskKey = "Configure_env", key = 'batch_size', default = 4)
bf16 = dbutils.jobs.taskValues.get(taskKey = "Configure_env", key = 'bf16', default = False)

sentence_transformer_dir = dbutils.jobs.taskValues.get(taskKey = "Configure_env", key = 'sentence_transformer_dir', default = '/dbfs/energex-llama2/models/sentence_transformers/deberta')

website_train_dataset = dbutils.jobs.taskValues.get(taskKey = "Configure_env", key = 'website_train_dataset')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure env for new cluster

# COMMAND ----------

# Install dependencies
!pip install --upgrade pip
!pip install deepspeed==0.9.1 py-cpuinfo==9.0.0
!pip install mlflow
!pip install accelerate>=0.16.0,<1 click>=8.0.4,<9 datasets>=2.10.0,<3 deepspeed>=0.8.3,<0.9 transformers[torch]>=4.28.1,<5 langchain>=0.0.139 torch>=1.13.1,<2
!pip install -U deepspeed --quiet
!pip install -U accelerate --quiet
!pip install --upgrade accelerate
!pip install --upgrade mlflow
!pip install databricks-cli
!pip install --upgrade accelerate
!pip install pyspark
!pip install --upgrade cloudpickle
!pip install transformers=4.35.2
!pip install --upgrade datasets

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
dbutils.fs.put("file:///root/.databrickscfg","[DEFAULT]\nhost=https://dbc-eb788f31-6c73.cloud.databricks.com\ntoken = "+token,overwrite=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

import sys
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import pandas as pd
import click
import torch
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    integrations,
    pipeline,
)
sys.path.append('/Workspace/Users/brett.smerdon@energyq.com.au/EQL-LLM/Jeff/scripts')

import transformers
import accelerate
from consts import (
    DEFAULT_INPUT_MODEL,
    DEFAULT_SEED,
    #PROMPT_WITH_INPUT_FORMAT,
    #PROMPT_NO_INPUT_FORMAT,
    END_KEY,
    INSTRUCTION_KEY,
    RESPONSE_KEY_NL,
    DEFAULT_TRAINING_DATASET,
    SINGLE_PROMPT_NO_INPUT_FORMAT,
    SINGLE_PROMPT_WITH_INPUT_FORMAT,
    END_INST_KEY_NL
)

import mlflow
from generate import generate_response, load_model_tokenizer_for_generate, InstructionTextGenerationPipeline
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.transformers import generate_signature_output
from mlflow.types import DataType, Schema, ColSpec
#from pyspark.sql import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training arguments

# COMMAND ----------

deepspeed = config_file_name
epochs = 2
per_device_train_batch_size = batch_size
per_device_eval_batch_size = batch_size
logging_steps = 20
save_steps = 300
save_total_limit = 20
eval_steps = 20
warmup_steps = 50
test_size = 25
lr = 2e-6
z_shot_model_dir = sentence_transformer_dir
train_dataset = website_train_dataset
seed = 42
local_rank = True
gradient_checkpointing = True

# COMMAND ----------

logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

logging.getLogger().setLevel(logging.INFO)
logging.getLogger("py4j").setLevel(logging.ERROR)
logger = logging.getLogger("Training")
logger.info("Started logger successfully")

# COMMAND ----------

def load_tokenizer(pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL) -> PreTrainedTokenizer:
    logger.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
    tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir='/dbfs/energex-llama2/models')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})
    return tokenizer


def load_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> AutoModelForCausalLM:
    logger.info(f"Loading model for {pretrained_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, cache_dir='/dbfs/energex-llama2/models', trust_remote_code=True, use_cache=False if gradient_checkpointing else True
    )
    return model

# COMMAND ----------

def get_model_tokenizer(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(pretrained_model_name_or_path, gradient_checkpointing=gradient_checkpointing)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer

# COMMAND ----------

model, tokenizer = get_model_tokenizer(
    pretrained_model_name_or_path=input_model, gradient_checkpointing=gradient_checkpointing
)

# Use the same max length that the model supports.  Fall back to 1024 if the setting can't be found.
# The configuraton for the length can be stored under different names depending on the model.  Here we attempt
# a few possible names we've encountered.
conf = model.config
max_length = None
for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
    max_length = getattr(model.config, length_setting, None)
    if max_length:
        logger.info(f"Found max lenth: {max_length}")
        break
if not max_length:
    max_length = 1024
    logger.info(f"Using default max length: {max_length}")
print(f"MAX LENGTH: {max_length}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dataset.from_spark caches the dataset. This example describes model training on the driver, so data must be made available to it. Additionally, since cache materialization is parallelized using Spark, the provided cache_dir must be accessible to all workers. To satisfy these constraints, cache_dir should be a Databricks File System (DBFS) root volume or mount point.

# COMMAND ----------

!cd /dbfs; mkdir cache; cd cache; mkdir train

# COMMAND ----------

def load_training_dataset(path_or_dataset: str) -> Dataset:
    
    logger.info(f"Loading dataset from {path_or_dataset}")
    df = spark.table(path_or_dataset)
    dataset = Dataset.from_spark(df, cache_dir="/dbfs/cache/train") #.select(range(0,20))

    logger.info("Found %d rows", dataset.num_rows)

    def _add_text(rec):
        instruction = rec["instruction"]
        response = rec["response"]
        context = rec.get("context")

        if not instruction:
            raise ValueError(f"Expected an instruction in: {rec}")

        if not response:
            raise ValueError(f"Expected a response in: {rec}")

        # For some instructions there is an input that goes along with the instruction, providing context for the
        # instruction.  For example, the input might be a passage from Wikipedia and the instruction says to extract
        # some piece of information from it.  The response is that information to extract.  In other cases there is
        # no input.  For example, the instruction might be open QA such as asking what year some historic figure was
        # born.
        if context:
            rec["text"] = SINGLE_PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
            print(rec["text"])
        else:
            rec["text"] = SINGLE_PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
            print(rec["text"])
        return rec

    dataset = dataset.map(_add_text)

    return dataset

# COMMAND ----------

def preprocess_batch(batch: Dict[str, List], tokenizer: LlamaTokenizer, max_length: int) -> dict:
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True
    )

# COMMAND ----------

def preprocess_dataset(tokenizer: LlamaTokenizer, max_length: int, training_dataset: str, seed=DEFAULT_SEED) -> Dataset:
    """Loads the training dataset and tokenizes it so it is ready for training.

    Args:
        tokenizer (LlamaTokenizer): Tokenizer tied to the model.
        max_length (int): Maximum number of tokens to emit from tokenizer.

    Returns:
        Dataset: HuggingFace dataset
    """

    dataset = load_training_dataset(training_dataset)

    logger.info("Preprocessing dataset")
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "context", "response", "text", "category"],
    )

    # Make sure we don't have any truncated records, as this would mean the end keyword is missing.
    logger.info("Processed dataset has %d rows", dataset.num_rows)
    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
    logger.info("Processed dataset has %d rows after filtering for truncated records", dataset.num_rows)

    logger.info("Shuffling dataset")
    dataset = dataset.shuffle(seed=seed)

    logger.info("Done preprocessing")

    return dataset

# COMMAND ----------

# # !huggingface-cli login
# # or using an environment variable
# !huggingface-cli login --token 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

# from transformers import (
#     AutoModelForCausalLM,
#     LlamaTokenizer,
#     DataCollatorForLanguageModeling,
#     PreTrainedTokenizer,
#     Trainer,
#     TrainingArguments,
#     set_seed,
#     integrations,
#     pipeline,
# )
# import sys
# sys.path.append('/Workspace/Users/brett.smerdon@energyq.com.au/EQL-LLM/Jeff/scripts')
# from consts import (
#     DEFAULT_INPUT_MODEL,
#     DEFAULT_SEED,
#     PROMPT_WITH_INPUT_FORMAT,
#     PROMPT_NO_INPUT_FORMAT,
#     END_KEY,
#     INSTRUCTION_KEY,
#     RESPONSE_KEY_NL,
#     DEFAULT_TRAINING_DATASET,
# )


# def load_tokenizer(pretrained_model_name_or_path: str = 'meta-llama/Llama-2-7b-chat-hf') -> PreTrainedTokenizer:
#     #logger.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
#     tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir='/dbfs/energex-llama2/models')
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})
#     return tokenizer


# def load_model(
#     pretrained_model_name_or_path: str = 'meta-llama/Llama-2-7b-chat-hf', *, gradient_checkpointing: bool = False
# ) -> AutoModelForCausalLM:
#     #logger.info(f"Loading model for {pretrained_model_name_or_path}")
#     model = AutoModelForCausalLM.from_pretrained(
#         pretrained_model_name_or_path, cache_dir='/dbfs/energex-llama2/models', trust_remote_code=True, use_cache=False if gradient_checkpointing else True
#     )
#     return model

# tknz = load_tokenizer('meta-llama/Llama-2-7b-chat-hf')
# mod = load_model('meta-llama/Llama-2-7b-chat-hf')


# COMMAND ----------

processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=max_length, seed=seed, training_dataset=train_dataset)

split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=seed)

logger.info("Train data size: %d", split_dataset["train"].num_rows)
logger.info("Test data size: %d", split_dataset["test"].num_rows)

# COMMAND ----------


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        end_inst_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):
            print(f"### Batch labels = {batch['labels'][i]}")
            print(f"### End token = {end_inst_token_ids[0]}")
            print(f"### Condition check = {np.where(batch['labels'][i] == end_inst_token_ids[0])}")
            print("--------------------------------------------------------------------------------------------")
            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == end_inst_token_ids[0])[0]:

                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {end_inst_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


# COMMAND ----------

data_collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
)

# enable fp16 if not bf16
fp16 = not bf16

if not dbfs_output_dir:
    logger.warn("Will NOT save to DBFS")

training_args = TrainingArguments(
    output_dir=local_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    fp16=fp16,
    bf16=bf16,
    learning_rate=lr,
    num_train_epochs=epochs,
    deepspeed=deepspeed,
    gradient_checkpointing=gradient_checkpointing,
    logging_dir=f"{local_output_dir}/runs",
    logging_strategy="steps",
    logging_steps=logging_steps,
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    load_best_model_at_end=False,
    report_to="tensorboard",
    disable_tqdm=False,
    remove_unused_columns=False,
    local_rank=local_rank,
    warmup_steps=warmup_steps,

)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Ensure logs are sent to databricks hosted mlflow

# COMMAND ----------

class MyCallback(integrations.MLflowCallback):
    "A TrainerCallback that sends the logs to MLflow"


# COMMAND ----------

logger.info("Instantiating Trainer")
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=data_collator,
    callbacks=[MyCallback]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define pyfunc load from mlflow

# COMMAND ----------

!pip install datetime
!pip install pytz

# COMMAND ----------

from datetime import datetime
import pytz

class EGX(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(
          context.artifacts['repository'], padding_side="left")

        config = transformers.AutoConfig.from_pretrained(
            context.artifacts['repository'], 
            trust_remote_code=True
        )
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'], 
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True)
        self.model.to(device='cuda')
        
        self.model.eval()
        self.zero_shot_pipeline = transformers.pipeline(
            task="zero-shot-classification",
            model = transformers.AutoModelForSequenceClassification.from_pretrained(context.artifacts['sentence_tx_dir']),
            tokenizer = transformers.AutoTokenizer.from_pretrained(context.artifacts['sentence_tx_dir']),
            device=0
        )

        self.website_page_names = {
            'prepare for outages': 'https://www.energex.com.au/outages/be-prepared-for-outages', 
            'electric vehicle': 'https://www.energex.com.au/manage-your-energy/smarter-energy/electric-vehicles-ev', 
            'hot water issues': 'https://www.energex.com.au/outages/hot-water-issues-cold-water', 
            'kids safety': 'https://www.energex.com.au/safety/kids-safety', 
            'manage your energy': 'https://www.energex.com.au/manage-your-energy', 
            'metering': 'https://www.energex.com.au/our-services/metering', 
            'network tariffs & prices"': 'https://www.energex.com.au/our-network/network-pricing-and-tariffs', 
            'outages finder': 'https://www.energex.com.au/outages/outage-finder', 
            'outages': 'https://www.energex.com.au/outages', 
            'safety at home': 'https://www.energex.com.au/safety/safety-at-home-or-work', 
            'safety': 'https://www.energex.com.au/safety', 
            'save money & electricity': 'https://www.energex.com.au/manage-your-energy/save-money-and-electricity', 
            'self-service': 'https://www.energex.com.au/our-services/self-service', 
            'solar power': 'https://www.energex.com.au/manage-your-energy/smarter-energy/solar-pv', 
            'storms and disasters': 'https://www.energex.com.au/outages/storms-and-disasters', 
            'working near powerlines': 'https://www.energex.com.au/safety/working-near-powerlines', 
            }

    def categorize_article(self, article: str) -> None:
        """
        This helper function defines the categories (labels) which the model must use to label articles.
        Note that our model was NOT fine-tuned to use these specific labels,
        but it "knows" what the labels mean from its more general training.

        This function then prints out the predicted labels alongside their confidence scores.
        """

        results = self.zero_shot_pipeline(article, candidate_labels=list(self.website_page_names.keys()))
        results.pop("sequence")
        rating = pd.DataFrame(results)['labels'][0]
        return rating

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        generated_text = []
        #for index, row in model_input.iterrows():
        prompt = model_input['prompt'].iloc[0] #row["prompt"]
        kwargs = {'torch_dtype': "auto"}
        generation_pipeline = InstructionTextGenerationPipeline(model=self.model, tokenizer=self.tokenizer, **kwargs)
        reponse = generation_pipeline(prompt)[0]["generated_text"]

        category = self.categorize_article(f"Instruction: {prompt}\nResponse: {reponse}")
        generated_text.append(reponse + '<@@@>' + self.website_page_names[category]) # Append zero shot response with response

        # Create a timezone-aware datetime object
        brisbane_time = datetime.now(pytz.timezone('Australia/Queensland'))
        formatted_time = brisbane_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')

        # df = spark.createDataFrame([(formatted_time, prompt, reponse)],["ts", "instruction", "response"]).collect()
        #df = spark.createDataFrame([(1, 2, 3)],["ts", "instruction", "response"]).collect()
        # # Append to audit table
        #df.write.format("delta").mode("append").saveAsTable("eql_llm.instructions.table_energex_LLM_history")

        return pd.Series(generated_text) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generates signatures and input example which will display as a guide for invoking endpoint once served

# COMMAND ----------

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_tokens")])

output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt":["what is ML?"], 
            "temperature": [0.5],
            "max_tokens": [2048]})
    
set_seed(seed)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training Procedure

# COMMAND ----------

logger.info("Training")

mlflow.set_registry_uri('databricks-uc')
CATALOG = "eql_llm"
SCHEMA = "prod-serve"
registered_model_name = f"{CATALOG}.{SCHEMA}.energex-LLAMA-7b"
mlflow.set_experiment("/Users/brett.smerdon@energyq.com.au/EQL-LLM/Common/Demo Notebooks/Fine-tune LLM")
# Train the model.
with mlflow.start_run() as run:

    trainer.train()

    logger.info(f"Saving Model to {local_output_dir}")
    trainer.save_model(output_dir=local_output_dir)

    logger.info("Completed Training")
    torch_version = torch.__version__.split("+")[0]
    #custom_pipeline = InstructionTextGenerationPipeline(model=trainer.model, tokenizer=trainer.tokenizer)#**pipeline_kwargs)
    mlflow.pyfunc.log_model(
        "model",
        python_model=EGX(),
        artifacts={'repository' : local_output_dir, 'sentence_tx_dir' : z_shot_model_dir},
        pip_requirements=[f"torch=={torch_version}", f"transformers=={transformers.__version__}", f"accelerate=={accelerate.__version__}", "einops", "sentencepiece"],
        code_path=['/Workspace/Users/brett.smerdon@energyq.com.au/EQL-LLM/Jeff/scripts/consts.py', '/Workspace/Users/brett.smerdon@energyq.com.au/EQL-LLM/Jeff/scripts/generate.py'],
        input_example=input_example,
        registered_model_name=registered_model_name,
        metadata={"task": "llm/v1/completions"},
        signature=signature
    )
