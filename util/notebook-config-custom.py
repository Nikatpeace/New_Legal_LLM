# Databricks notebook source
# MAGIC %pip install mlflow

# COMMAND ----------

if 'config' not in locals():
  config = {}

# COMMAND ----------

# DBTITLE 1,Set document path
#config['kb_documents_path'] = "/Users/nikhil.chandna@databricks.com/demos/llm"
#config['vector_store_path'] = '/dbfs/tmp/qabot/vector_store' # /dbfs/... is a local file system representation
#config['test_vector_store_path'] = '/dbfs/tmp/qabot/temp/test_vector_store' # /dbfs/... is a local file system representation

# COMMAND ----------

# DBTITLE 1,Create database
#config['database_name'] = 'nik_demo2_catalog.nik_qabot'

# create database if not exists
#_ = spark.sql(f"create database if not exists {config['database_name']}")

# set current datebase context
#_ = spark.catalog.setCurrentDatabase(config['database_name'])

# COMMAND ----------

# DBTITLE 1,Set Environmental Variables for tokens
import os

os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("nik_qa_bot", "new_key")

# COMMAND ----------

# DBTITLE 1,mlflow settings
import mlflow
config['registered_model_name'] = 'nik_VS_legal_llm_qabot'
config['model_uri'] = f"models:/{config['registered_model_name']}/production"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
_ = mlflow.set_experiment('/Users/{}/{}'.format(username, config['registered_model_name']))

# COMMAND ----------

# DBTITLE 1,Set OpenAI model configs
config['openai_embedding_model'] = 'databricks-bge-large-en'
config['openai_chat_model'] = "gpt-3.5-turbo"
config['system_message_template'] = """You are a helpful assistant built by Databricks, you are good at helping to answer a question based on the context provided, the context is a document. If the context does not provide enough relevant information to determine the answer, just say I don't know. If the context is irrelevant to the question, just say I don't know. If you did not find a good answer from the context, just say I don't know. If the query doesn't form a complete question, just say I don't know. If there is a good answer from the context, try to summarize the context to answer the question."""
config['human_message_template'] = """Given the context: {context}. Answer the question {question}."""
config['temperature'] = 0.15

# COMMAND ----------

# DBTITLE 1,Set evaluation config
config["eval_dataset_path"]= "./data/eval_data.tsv"

# COMMAND ----------

# DBTITLE 1,Set deployment configs
config['openai_key_secret_scope'] = "nik_qa_bot" # See `./RUNME` notebook for secret scope instruction - make sure it is consistent with the secret scope name you actually use 
config['openai_key_secret_key'] = "new_key" # See `./RUNME` notebook for secret scope instruction - make sure it is consistent with the secret scope key name you actually use
config['serving_endpoint_name'] = "VS-llm-qabot-endpoint"
