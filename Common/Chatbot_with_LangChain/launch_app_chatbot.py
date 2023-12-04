# Databricks notebook source
# MAGIC %sh
# MAGIC /databricks/python/bin/pip install fastapi streamlit uvicorn langchain accelerate torch

# COMMAND ----------

import os
app_port = 8501
os.environ['DB_APP_PORT'] = f'{app_port}'

cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")

workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
org_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")

# For AWS - Azure can have a different path and corporate workspaces can have their own uri
proxy_prefix = f'dbc-dp-{org_id}.cloud.databricks.com'
endpoint_url = f"https://{proxy_prefix}/driver-proxy/o/{org_id}/{cluster_id}/{app_port}/"
print(f"Access this API at {endpoint_url}")

# COMMAND ----------

!streamlit run app.py

# COMMAND ----------


