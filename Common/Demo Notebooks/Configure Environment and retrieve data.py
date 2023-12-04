# Databricks notebook source
# MAGIC %md
# MAGIC # Set up environment and read data

# COMMAND ----------

# Install dependencies
!pip install --upgrade pip
!pip install deepspeed==0.9.1 py-cpuinfo==9.0.0
!pip install mlflow
!pip install accelerate>=0.16.0 click>=8.0.4 datasets>=2.10.0 deepspeed>=0.8.3 transformers[torch]>=4.28.1 langchain>=0.0.139 torch>=1.13.1
!pip install -U deepspeed --quiet
!pip install -U accelerate --quiet
!pip install --upgrade accelerate
!pip install --upgrade mlflow

# COMMAND ----------

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)

# COMMAND ----------

import os
import re
from datetime import datetime

working_dir = '/Workspace/Repos/jeff.mao@energyq.com.au/eql-llm-cup-2023/Common/Demo Notebooks'

timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
model_name = "llama2"

# experiment_id = dbutils.widgets.get("experiment_id")
experiment_id = "energex-0.1"
# input_model = dbutils.widgets.get("input_model")
input_model = 'meta-llama/Llama-2-7b-chat-hf'
if experiment_id:
    experiment_id = re.sub(r"\s+", "_", experiment_id.strip())
    model_name = f"{model_name}__{experiment_id}"

checkpoint_dir_name = f"{model_name}__{timestamp}"

llama2_training_dir_name = "energex-llama2/training"

# Use the local training root path if it was provided.  Otherwise try to find a sensible default.
# local_training_root = dbutils.widgets.get("local_training_root")
local_training_root = '/dbfs/energex-llama2/training/local/'
if not local_training_root:
    # Use preferred path when working in a Databricks cluster if it exists.
    if os.path.exists("/local_disk0"):
        local_training_root = os.path.join("/local_disk0", llama2_training_dir_name)
    # Otherwise use the home directory.
    else:
        local_training_root = os.path.join(os.path.expanduser('~'), llama2_training_dir_name)

# dbfs_output_root = dbutils.widgets.get("dbfs_output_root")
dbfs_output_root = '/dbfs/energex-llama2/training/output/'
if not dbfs_output_root:
    dbfs_output_root = f"/dbfs/{llama2_training_dir_name}"

os.makedirs(local_training_root, exist_ok=True)
os.makedirs(dbfs_output_root, exist_ok=True)

local_output_dir = os.path.join(local_training_root, checkpoint_dir_name)
dbfs_output_dir = os.path.join(dbfs_output_root, checkpoint_dir_name)
tensorboard_display_dir = f"{local_output_dir}/runs"

print(f"Local Output Dir: {local_output_dir}")
print(f"DBFS Output Dir: {dbfs_output_dir}")
print(f"Tensorboard Display Dir: {tensorboard_display_dir}")

# pick an appropriate config file
gpu_family = "a10"
config_file_name = os.path.join(working_dir, f"configs/{gpu_family}_config.json")
# config_file_name = f"{gpu_family}_config.json"
# deepspeed_config = os.path.join(os.getcwd(), "config", config_file_name)
# deepspeed_config = '/Workspace/Users/brett.smerdon@energyq.com.au/EQL-LLM/Jeff/configs/ds_config.json'
print(f"Deepspeed config file: {config_file_name}")

# configure the batch_size
batch_size = 3
if gpu_family == "a10":
    batch_size = 4
elif gpu_family == "a100":
    batch_size = 6

# configure num_gpus, if specified
num_gpus_flag = ""

num_gpus = 1
if num_gpus:
    num_gpus = int(num_gpus)
    num_gpus_flag = f"--num_gpus={num_gpus}"

if gpu_family == "v100":
    bf16_flag = "--bf16 false"
else:
    bf16_flag = "--bf16 true"
bf16 = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define dataset

# COMMAND ----------

website_train_dataset = 'eql_llm.instructions.train_dataset_energex_550'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save zero shot classifier to be used as a mlflow artifact

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
saved_model_dir = '/dbfs/energex-llama2/models/sentence_transformers/'
sentence_transformer_model_name = 'deberta'

sentence_transformer_dir = saved_model_dir + sentence_transformer_model_name

sent_tx_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-large', cache_dir=sentence_transformer_dir, trust_remote_code=True)
sent_tx_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-large', cache_dir=sentence_transformer_dir, trust_remote_code=True)

sent_tx_model.save_pretrained(sentence_transformer_dir, from_pt=True)
sent_tx_tokenizer.save_pretrained(sentence_transformer_dir, from_pt=True)

del sent_tx_model, sent_tx_tokenizer
print(sentence_transformer_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set task parameters for dependent steps in the workflow

# COMMAND ----------

dbutils.jobs.taskValues.set(key='input_model', value=input_model)
dbutils.jobs.taskValues.set(key='config_file_name', value=config_file_name)
dbutils.jobs.taskValues.set(key='local_output_dir', value=local_output_dir) 
dbutils.jobs.taskValues.set(key='dbfs_output_dir', value=dbfs_output_dir) 
dbutils.jobs.taskValues.set(key='batch_size', value=batch_size)
dbutils.jobs.taskValues.set(key='bf16', value=bf16)
dbutils.jobs.taskValues.set(key='sentence_transformer_dir', value=sentence_transformer_dir)
dbutils.jobs.taskValues.set(key='website_train_dataset', value=website_train_dataset) 
