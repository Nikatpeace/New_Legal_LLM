# Databricks notebook source
# MAGIC %md The purpose of this notebook is to access and prepare our data for use with the QA Bot accelerator.  This notebook is available at https://github.com/databricks-industry-solutions/diy-llm-qa-bot.

# COMMAND ----------

# MAGIC %md
# MAGIC #This is the custom code
# MAGIC

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch
# MAGIC %pip install --upgrade langchain 
# MAGIC %pip install tiktoken==0.4.0 openai==0.27.6 typing-inspect==0.8.0 typing_extensions==4.5.0
# MAGIC %pip install langchain-core
# MAGIC %pip install typing-extensions --upgrade
# MAGIC %pip install gradio
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

#df_text_chunked_small = spark.read.table('nik_demo2_catalog.nik_qabot.legal_data_45k_text_chunked_small')

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# COMMAND ----------

help(VectorSearchClient)

# COMMAND ----------

vsc.list_indexes('one-env-shared-endpoint-1')

# COMMAND ----------

index = vsc.get_index('one-env-shared-endpoint-1', 'nik_demo2_catalog.nik_qabot.legal_data_45k_text_chunked_small_vs_index')

# COMMAND ----------

#vector_search_endpoint_name = "nik_vector-search-demo-endpoint"

#vsc.create_endpoint(
#    name=vector_search_endpoint_name,
#    endpoint_type="STANDARD"
#)

# COMMAND ----------

# Vector index
#vs_index = "legal_data_45k_text_chunked_small"
#vs_index_fullname = f"nik_demo2_catalog.nik_qabot.legal_data_45k_text_chunked_small_vs_index"

#embedding_model_endpoint = "databricks-bge-large-en"

# COMMAND ----------

# MAGIC %sql
# MAGIC --ALTER TABLE nik_demo2_catalog.nik_qabot.legal_data_45k_text_chunked_small SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

#index = vsc.create_delta_sync_index(
#  endpoint_name=vector_search_endpoint_name,
#  source_table_name="nik_demo2_catalog.nik_qabot.legal_data_45k_text_chunked_small",
#  index_name=vs_index_fullname,
#  pipeline_type='TRIGGERED',
 # primary_key="index",
  #embedding_source_column="text",
  #embedding_model_endpoint_name=embedding_model_endpoint
#)
#index.describe()

# COMMAND ----------


index.describe()

# COMMAND ----------

all_columns = spark.table("nik_demo2_catalog.nik_qabot.legal_data_45k_text_chunked_small").columns

results = index.similarity_search(
  query_text="crucial evidence in a murder case",
  columns=all_columns,
  num_results=2)

results

# COMMAND ----------

#%pip install --upgrade langchain

# COMMAND ----------

from langchain.schema import Document
from typing import List

def convert_vector_search_to_documents(results) -> List[Document]:
  column_names = []
  for column in results["manifest"]["columns"]:
      column_names.append(column)

  langchain_docs = []
  for item in results["result"]["data_array"]:
      metadata = {}
      score = item[-1]
      # print(score)
      i = 1
      for field in item[1:-1]:
          # print(field + "--")
          metadata[column_names[i]["name"]] = field
          i = i + 1
      doc = Document(page_content=item[0], metadata=metadata)  # , 9)
      langchain_docs.append(doc)
  return langchain_docs

langchain_docs = convert_vector_search_to_documents(results)

langchain_docs

# COMMAND ----------

docs = langchain_docs
for doc in docs: 
  print(doc,'\n') 

# COMMAND ----------

# MAGIC %run "./util/notebook-config-custom"

# COMMAND ----------

import re
import time
import pandas as pd
import mlflow
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import BaseRetriever
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain import LLMChain
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
#from langchain.vectorstores import DatabricksVectorSearch
from langchain_community.vectorstores import DatabricksVectorSearch
from typing import List
import itertools
import gradio as gr
import requests
import os
from gradio.themes.utils import sizes

# COMMAND ----------

#question = "Can you provide details for a property dispute case?"
#question = "Can you provide examples of evidence provided by the prosecutor?"
question = "Are there any Murder cases?"
#question = "What was the most important evidence considered in murder cases?"
#question = "What were the sections used by the prosecution in murder cases?"

# COMMAND ----------

# define system-level instructions
system_message_prompt = SystemMessagePromptTemplate.from_template(config['system_message_template'])

# define human-driven instructions
human_message_prompt = HumanMessagePromptTemplate.from_template(config['human_message_template'])

# combine instructions into a single prompt
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# define model to respond to prompt
llm = ChatOpenAI(model_name=config['openai_chat_model'], temperature=config['temperature'])

# combine prompt and model into a unit of work (chain)
qa_chain = LLMChain(
  llm = llm,
  prompt = chat_prompt
  )

# COMMAND ----------

# for each provided document
for doc in docs:

  # get document text
  text = doc.page_content

  # generate a response
  output = qa_chain.generate([{'context': text, 'question': question}])
 
  # get answer from results
  generation = output.generations[0][0]
  answer = generation.text

  # display answer
  if answer is not None:
    print(f"Question: {question}", '\n', f"Answer: {answer}")
    break
  else:
    break

# COMMAND ----------

class QABot():


  def __init__(self, llm, retriever, prompt):
    self.llm = llm
    self.retriever = retriever
    self.prompt = prompt
    self.qa_chain = LLMChain(llm = self.llm, prompt=prompt)
    #self.abbreviations = { # known abbreviations we want to replace
    #  "DBR": "Databricks Runtime",
    #  "ML": "Machine Learning",
    #  "UC": "Unity Catalog",
    #  "DLT": "Delta Live Table",
    #  "DBFS": "Databricks File Store",
    #  "HMS": "Hive Metastore",
    #  "UDF": "User Defined Function"
    #  } 


  def _is_good_answer(self, answer):

    ''' check if answer is a valid '''

    result = True # default response

   # badanswer_phrases = [ # phrases that indicate model produced non-answer
   #   "no information", "no context", "don't know", "no clear answer", "sorry", 
   #   "no answer", "no mention", "reminder", "context does not provide", "no helpful answer", 
   #   "given context", "no helpful", "no relevant", "no question", "not clear",
   #   "don't have enough information", " does not have the relevant information", "does not seem to be directly related"
   #   ]

    badanswer_phrases = [ # phrases that indicate model produced non-answer
      "no information", "no context", "don't know", "no clear answer", "sorry", 
      "no answer", "no mention", "reminder", "context does not provide", "no helpful answer",
      "no helpful", "no relevant", "no question", "not clear",
      "don't have enough information", " does not have the relevant information", "does not seem to be directly related"
      ]
    
    if answer is None: # bad answer if answer is none
      results = False
    else: # bad answer if contains badanswer phrase
      for phrase in badanswer_phrases:
        if phrase in answer.lower():
          result = False
          break
    
    return result


  def _get_answer(self, context, question, timeout_sec=60):

    '''' get answer from llm with timeout handling '''

    # default result
    result = None

    # define end time
    end_time = time.time() + timeout_sec

    # try timeout
    while time.time() < end_time:

      # attempt to get a response
      try: 
        result =  qa_chain.generate([{'context': context, 'question': question}])
        break # if successful response, stop looping

      # if rate limit error...
      except openai.error.RateLimitError as rate_limit_error:
        if time.time() < end_time: # if time permits, sleep
          time.sleep(2)
          continue
        else: # otherwise, raiser the exception
          raise rate_limit_error

      # if other error, raise it
      except Exception as e:
        print(f'LLM QA Chain encountered unexpected error: {e}')
        raise e

    return result


  def get_answer(self, question):
    ''' get answer to provided question '''

    # default result
    result = {'answer':None, 'source':None, 'output_metadata':None}

    # remove common abbreviations from question
   # for abbreviation, full_text in self.abbreviations.items():
   #   pattern = re.compile(fr'\b({abbreviation}|{abbreviation.lower()})\b', re.IGNORECASE)
   #   question = pattern.sub(f"{abbreviation} ({full_text})", question)

    # get relevant documents
    docs = self.retriever.get_relevant_documents(question)

    # for each doc ...
    for doc in docs:

      # get key elements for doc
      text = doc.page_content
      #source = doc.metadata['source']
      if 'source' in doc.metadata:
                source = doc.metadata['source']
      else:
                source = 'unknown source'

      # get an answer from llm
      output = self._get_answer(text, question)
 
      # get output from results
      generation = output.generations[0][0]
      answer = generation.text
      output_metadata = output.llm_output

      # assemble results if not no_answer
      if self._is_good_answer(answer):
        result['answer'] = answer
        result['source'] = source
        result['output_metadata'] = output_metadata
        break # stop looping if good answer
      
    return result

# COMMAND ----------

class DatabricksVectorRetriever(BaseRetriever):
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def _get_relevant_documents(self, query: str) -> List[Document]:
        similar_documents = self.vectorstore.get_relevant_documents(query)
        return [Document(page_content=document) for document in similar_documents]

# COMMAND ----------


def get_retriever(persist_dir: str = None):
    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        index, text_column="text", embedding=config['openai_embedding_model']
    )
    return vectorstore.as_retriever(search_kwargs={"k":2})




# COMMAND ----------

# test our retriever
vectorstore = get_retriever()
similar_documents = vectorstore.get_relevant_documents("Are there any murder cases")
#similar_documents = vectorstore.get_relevant_documents("Are there any murder cases", filter={"client":"google"})
print(f"Relevant documents: {similar_documents}")

# COMMAND ----------

# instantiate bot object
qabot = QABot(llm, vectorstore, chat_prompt)

# get response to question
qabot.get_answer(question) 

# COMMAND ----------

class MLflowQABot(mlflow.pyfunc.PythonModel):

  def __init__(self, llm, retriever, chat_prompt):
    self.qabot = QABot(llm, vectorstore, chat_prompt)

  def predict(self, context, inputs):
    questions = list(inputs['question'])

    # return answer
    return [self.qabot.get_answer(q) for q in questions]

# COMMAND ----------

# instantiate mlflow model
model = MLflowQABot(llm, vectorstore, chat_prompt)

# persist model to mlflow
with mlflow.start_run():
  _ = (
    mlflow.pyfunc.log_model(
      python_model=model,
      extra_pip_requirements=['langchain', 'tiktoken==0.4.0', 'openai==0.27.6', 'typing-inspect==0.8.0', 'typing_extensions==4.5.0','databricks-vectorsearch'],
      artifact_path='model',
      registered_model_name=config['registered_model_name']
      )
    )


# COMMAND ----------

# connect to mlflow 
client = mlflow.MlflowClient()

# identify latest model version
latest_version = client.get_latest_versions(config['registered_model_name'], stages=['None'])[0].version

# move model into production
client.transition_model_version_stage(
    name=config['registered_model_name'],
    version=latest_version,
    stage='Production',
    archive_existing_versions=True
)

# COMMAND ----------

# retrieve model from mlflow
model = mlflow.pyfunc.load_model(f"models:/{config['registered_model_name']}/Production")

# assemble question input
queries = pd.DataFrame({'question':[
  "Can you provide details for a property dispute case?"
]})

# get a response
model.predict(queries)

# COMMAND ----------

print(model)

# COMMAND ----------

latest_version = mlflow.MlflowClient().get_latest_versions(config['registered_model_name'], stages=['Production'])[0].version

# COMMAND ----------

print(mlflow.MlflowClient().get_latest_versions(config['registered_model_name'], stages=['Production']))

# COMMAND ----------


served_models = [
    {
      "name": "current",
      "model_name": config['registered_model_name'],
      "model_version": latest_version,
      "workload_size": "Small",
      "scale_to_zero_enabled": "true",
      "env_vars": [{
        "env_var_name": "OPENAI_API_KEY",
        "secret_scope": config['openai_key_secret_scope'],
        "secret_key": config['openai_key_secret_key'],
      }]
    }
]
traffic_config = {"routes": [{"served_model_name": "current", "traffic_percentage": "100"}]}

# COMMAND ----------

def endpoint_exists():
  """Check if an endpoint with the serving_endpoint_name exists"""
  url = f"https://{serving_host}/api/2.0/serving-endpoints/{config['serving_endpoint_name']}"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  response = requests.get(url, headers=headers)
  return response.status_code == 200

def wait_for_endpoint():
  """Wait until deployment is ready, then return endpoint config"""
  headers = { 'Authorization': f'Bearer {creds.token}' }
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints/{config['serving_endpoint_name']}"
  response = requests.request(method='GET', headers=headers, url=endpoint_url)
  while response.json()["state"]["ready"] == "NOT_READY" or response.json()["state"]["config_update"] == "IN_PROGRESS" : # if the endpoint isn't ready, or undergoing config update
    print("Waiting 30s for deployment or update to finish")
    time.sleep(30)
    response = requests.request(method='GET', headers=headers, url=endpoint_url)
    response.raise_for_status()
  return response.json()

def create_endpoint():
  """Create serving endpoint and wait for it to be ready"""
  print(f"Creating new serving endpoint: {config['serving_endpoint_name']}")
  endpoint_url = f'https://{serving_host}/api/2.0/serving-endpoints'
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = {"name": config['serving_endpoint_name'], "config": {"served_models": served_models}}
  json_bytes = json.dumps(request_data).encode('utf-8')
  response = requests.post(endpoint_url, data=json_bytes, headers=headers)
  response.raise_for_status()
  wait_for_endpoint()
  displayHTML(f"""Created the <a href="/#mlflow/endpoints/{config['serving_endpoint_name']}" target="_blank">{config['serving_endpoint_name']}</a> serving endpoint""")
  
def update_endpoint():
  """Update serving endpoint and wait for it to be ready"""
  print(f"Updating existing serving endpoint: {config['serving_endpoint_name']}")
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints/{config['serving_endpoint_name']}/config"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = { "served_models": served_models, "traffic_config": traffic_config }
  json_bytes = json.dumps(request_data).encode('utf-8')
  response = requests.put(endpoint_url, data=json_bytes, headers=headers)
  response.raise_for_status()
  wait_for_endpoint()
  displayHTML(f"""Updated the <a href="/#mlflow/endpoints/{config['serving_endpoint_name']}" target="_blank">{config['serving_endpoint_name']}</a> serving endpoint""")

# COMMAND ----------

from mlflow.utils.databricks_utils import get_databricks_host_creds
import mlflow
import requests
import json
import time

# COMMAND ----------

serving_host = spark.conf.get("spark.databricks.workspaceUrl")
creds = get_databricks_host_creds()

# COMMAND ----------

# gather other inputs the API needs
serving_host = spark.conf.get("spark.databricks.workspaceUrl")
creds = get_databricks_host_creds()

# kick off endpoint creation/update
if not endpoint_exists():
  create_endpoint()
else:
  update_endpoint()

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

endpoint_url = f"""https://{serving_host}/serving-endpoints/{config['serving_endpoint_name']}/invocations"""


def create_tf_serving_json(data):
    return {
        "inputs": {name: data[name].tolist() for name in data.keys()}
        if isinstance(data, dict)
        else data.tolist()
    }


def score_model(dataset):
    url = endpoint_url
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
    }
    ds_dict = (
        {"dataframe_split": dataset.to_dict(orient="split")}
        if isinstance(dataset, pd.DataFrame)
        else create_tf_serving_json(dataset)
    )
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    print(headers)
    print(data_json)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )

    return response.json()

# COMMAND ----------

print(endpoint_url)

# COMMAND ----------

# assemble question input
queries = pd.DataFrame({'question':[
  "Can you provide details for a property dispute case?"
]})

score_model( 
   queries
    )

# COMMAND ----------

# MAGIC %pip install gradio

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

#!pip install fastapi==0.104.1 typing_extensions==4.8.0 gradio==3.41.0


# COMMAND ----------

# MAGIC %pip install typing-extensions --upgrade
# MAGIC

# COMMAND ----------

endpoint_url = f"""https://{serving_host}/serving-endpoints/{config['serving_endpoint_name']}/invocations"""

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade

# COMMAND ----------

# MAGIC %sh
# MAGIC ./lib

# COMMAND ----------

from lib.notebook import gradio_launch

# COMMAND ----------

# MAGIC %run
# MAGIC ./lib

# COMMAND ----------

import sys;print(sys.path)

# COMMAND ----------


from lib.notebook import gradio_launch

 

def respond(message, history):

    if len(message.strip()) == 0:
        return "ERROR the question should not be empty"

    #local_token = os.getenv('API_TOKEN')
    local_token = {creds.token}
    #local_endpoint = os.getenv('API_ENDPOINT')
    local_endpoint = endpoint_url

    if local_token is None or local_endpoint is None:
        return "ERROR missing env variables"

    # Add your API token to the headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {local_token}'
    }

    #prompt = list(itertools.chain.from_iterable(history))
    #prompt.append(message)
    #q = {"inputs": [prompt]}
    q = {"inputs": [message]}
    try:
        response = requests.post(
            local_endpoint, json=q, headers=headers, timeout=100)
        response_data = response.json()
        #print(response_data)
        response_data=response_data["predictions"][0]
        #print(response_data)

    except Exception as error:
        response_data = f"ERROR status_code: {type(error).__name__}"
        # + str(response.status_code) + " response:" + response.text

    # print(response.json())
    return response_data


theme = gr.themes.Soft(
    text_size=sizes.text_sm,radius_size=sizes.radius_sm, spacing_size=sizes.spacing_sm,
)


demo = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(show_label=False, container=False, show_copy_button=True, bubble_full_width=True),
    textbox=gr.Textbox(placeholder="Ask me a question",
                       container=False, scale=7),
    title="Databricks LLM RAG demo - Chat with DBRX Databricks model serving endpoint",
    description="This chatbot is a demo example for the dbdemos llm chatbot. <br>This content is provided as a LLM RAG educational example, without support. It is using DBRX, can hallucinate and should not be used as production content.<br>Please review our dbdemos license and terms for more details.",
    examples=[["What is DBRX?"],
              ["How can I start a Databricks cluster?"],
              ["What is a Databricks Cluster Policy?"],
              ["How can I track billing usage on my workspaces?"],],
    cache_examples=False,
    theme=theme,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)

if __name__ == "__main__":
    gradio_launch(demo)

# COMMAND ----------

import gradio as gr
def question_answer(context, question):
    pass  # Implement your question-answering model here...

gr.Interface(fn=question_answer, inputs=["text", "text"], 
             outputs=["textbox", "text"]).launch()

# COMMAND ----------

print(local_endpoint)

# COMMAND ----------

# MAGIC %pip show typing-extensions
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##Custom code ends here

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |
