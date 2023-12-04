# Databricks notebook source
# MAGIC %md
# MAGIC # Launching Streamlit on Databricks
# MAGIC Databricks can support applications
# MAGIC
# MAGIC the application will run on the driver node through the driver proxy

# COMMAND ----------

# DBTITLE 1,Install your libraries into the root env
!pip install streamlit pillow

# COMMAND ----------

dbutils.library.restartPython() 


# COMMAND ----------

# DBTITLE 1,Finding your app path
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

# DBTITLE 1,Starting your application
os.environ["DATABRICKS_TOKEN"] = "<insert your token here>"

!streamlit run eql-model-query-profanity-chat.py
#!streamlit run eql-model-query.py
#!streamlit run /Volumes/eql_llm/app_serving/eql_app_serving/eql-model-query_volume.py
# !streamlit run eql-model-query-with-zero-shot.py

#/Volumes/eql_llm/app_serving/eql_app_serving/website/outages.htm
