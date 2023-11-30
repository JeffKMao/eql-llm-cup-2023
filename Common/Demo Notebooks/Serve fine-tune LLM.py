# Databricks notebook source
# MAGIC %md
# MAGIC # Serves latest model to Energex-L2M1 endpoint

# COMMAND ----------

from mlflow import MlflowClient
import mlflow

CATALOG = "eql_llm"
SCHEMA = "prod-serve"
registered_model_name = f"{CATALOG}.{SCHEMA}.energex-LLAMA-7b"

# Name of the registered MLflow model
model_name = registered_model_name
 
# Get the latest version of the MLflow model
mlflow.set_registry_uri('databricks-uc')
client = MlflowClient()
model_metadata = client.search_model_versions(f"name = '{model_name}'")
latest_model_version = model_metadata[0].version
model_version = int(latest_model_version)

# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
workload_type = "MULTIGPU_MEDIUM" 
 
# Specify the compute scale-out size(Small, Medium, Large, etc.)
workload_size = "Medium" 
 
# Specify Scale to Zero(only supported for CPU endpoints)
scale_to_zero = False 

# Set the name of the MLflow endpoint
endpoint_name = f"Energex-L2M1-prod"
 
# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

import requests
import json
 
data = {
    "name": endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": model_name,
                "model_version": model_version,
                "workload_size": workload_size,
                "scale_to_zero_enabled": scale_to_zero,
                "workload_type": workload_type,
            }
        ]
    },
}
 
headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

endpoint_url = f"{API_ROOT}/api/2.0/serving-endpoints"
url = f"{endpoint_url}/{endpoint_name}"
r = requests.get(url, headers=headers)

# Create endpoint if Energex-L2M1 doesn't already exist
if "RESOURCE_DOES_NOT_EXIST" in r.text:  
    print("Creating this new endpoint: ", f"{endpoint_url}/{endpoint_name}/invocations")
    response = requests.post(url=endpoint_url, headers=headers, json=data)
else:
    # Update config of existing endpoint
    print("Updating existing endpoint: ", f"{endpoint_url}/{endpoint_name}/invocations")
    response = requests.put(url=f"{endpoint_url}/{endpoint_name}/config", headers=headers, json=data['config'])


 
print(json.dumps(response.json(), indent=4))
