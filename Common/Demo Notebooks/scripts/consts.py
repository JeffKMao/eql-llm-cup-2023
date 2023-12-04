DEFAULT_INPUT_MODEL = "meta-llama/Llama-2-7b-chat-hf"
SUGGESTED_INPUT_MODELS = [
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "EleutherAI/gpt-j-6B",
    "databricks/dolly-v2-3b",
    "databricks/dolly-v2-7b",
    "databricks/dolly-v2-12b"
]
DEFAULT_TRAINING_DATASET = "databricks/databricks-dolly-15k" # Not used

INTRO_BLURB = ("You are a helpful, respectful and honest assistant from a large electricity distribution network service provider who is an expert on energy efficiency, electricity tariffs, solar energy installations, power outages and electricity safety. Always answer your customer's questions as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Do not repeat yourself.")

# (
#     "Below is an instruction that describes a task. Write a response that appropriately completes the request."
# )
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = '</s>'
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
DEFAULT_SEED = 42

# This is a training prompt that does not contain an input string.  The instruction by itself has enough information
# to respond.  For example, the instruction might ask for the year a historic figure was born.
SINGLE_PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

# This is a training prompt that contains an input string that serves as context for the instruction.  For example,
# the input might be a passage from Wikipedia and the intruction is to extract some information from it.
SINGLE_PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input="{input}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

PROMPT_FOR_GENERATION_FORMAT = """{intro}

{instruction} {response_key}
""".format(
    intro=INTRO_BLURB,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)