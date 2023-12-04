# Databricks notebook source
# MAGIC %md
# MAGIC # Run LLMs locally to ask questions about your documents

# COMMAND ----------

!pip install langchain
!pip install transformers
!pip install sentence_transformers
!pip install xformers
!pip install unstructured
#!pip install chromadb
!pip install chromadb==0.4.15

# COMMAND ----------

!pip list | grep chroma


# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts import PromptTemplate

# COMMAND ----------

hf_embeddings = HuggingFaceEmbeddings()

# COMMAND ----------

model_id = 'databricks/dolly-v2-3b' # find a suited text generation model on huggingface
llm = HuggingFacePipeline.from_model_id(model_id=model_id, device=0, task="text-generation", model_kwargs={"temperature":0, "max_length":2048})

# COMMAND ----------

#!curl https://en.wikipedia.org/wiki/Solar_cell > wiki_solar.html  # imported manually

# COMMAND ----------

#url = '.wiki_solar.html'
url = "/Workspace/Users/brett.smerdon@energyq.com.au/EQL-LLM/Brett/wiki_solar.html"
loader = UnstructuredHTMLLoader(url)

# COMMAND ----------

index = VectorstoreIndexCreator(embedding=hf_embeddings).from_loaders([loader])

# COMMAND ----------

url = "/Workspace/Users/brett.smerdon@energyq.com.au/EQL-LLM/Brett/website/energex.htm"
loader = UnstructuredHTMLLoader(url)
index = VectorstoreIndexCreator(embedding=hf_embeddings).from_loaders([loader])

# COMMAND ----------

url = "/Workspace/Users/brett.smerdon@energyq.com.au/EQL-LLM/Brett/website/safety.htm"
loader = UnstructuredHTMLLoader(url)
index = VectorstoreIndexCreator(embedding=hf_embeddings).from_loaders([loader])

# COMMAND ----------

url = "/Workspace/Users/brett.smerdon@energyq.com.au/EQL-LLM/Brett/website/outages.htm"
loader = UnstructuredHTMLLoader(url)
index = VectorstoreIndexCreator(embedding=hf_embeddings).from_loaders([loader])

# COMMAND ----------

url = "/Workspace/Users/brett.smerdon@energyq.com.au/EQL-LLM/Brett/website/metering.htm"
loader = UnstructuredHTMLLoader(url)
index = VectorstoreIndexCreator(embedding=hf_embeddings).from_loaders([loader])

# COMMAND ----------

url = "/Workspace/Users/brett.smerdon@energyq.com.au/EQL-LLM/Brett/website/manage-your-energy.htm"
loader = UnstructuredHTMLLoader(url)
index = VectorstoreIndexCreator(embedding=hf_embeddings).from_loaders([loader])

# COMMAND ----------

url = "/Workspace/Users/brett.smerdon@energyq.com.au/EQL-LLM/Brett/website/our-services.htm"
loader = UnstructuredHTMLLoader(url)
index = VectorstoreIndexCreator(embedding=hf_embeddings).from_loaders([loader])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ask it questions...

# COMMAND ----------

question = 'How should I teach my children electrical safety?'
index.query(question=question, llm=llm)

# COMMAND ----------

question = 'what is Energex?'
index.query(question=question, llm=llm)

# COMMAND ----------

question = 'How do I find out more about outages?'
index.query(question=question, llm=llm)

# COMMAND ----------

question = 'how do I contact energex?'
index.query(question=question, llm=llm)

# COMMAND ----------

question = 'What is the benefit of installing solar panels?'
index.query(question=question, llm=llm)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Trying  it with a Template

# COMMAND ----------

template = """Energex is the electricity distributor in South East Queensland. You are an employee of Energex.

{context}

Answer only with information from the energex.com.au website.

Question: {question}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
print(
    prompt.format(
        context="A customer is asking you a question.",
        question="How to stay safe near electricity?",
    )
)

# COMMAND ----------

import chromadb
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader

# COMMAND ----------

#loader = WebBaseLoader("https://energex.com.au/")

url = "/Workspace/Users/brett.smerdon@energyq.com.au/EQL-LLM/Brett/website/our-services.htm"
loader = UnstructuredHTMLLoader(url)

documents = loader.load()
len(documents)

# COMMAND ----------


embeddings = hf_embeddings
db = Chroma.from_documents(documents, embeddings)

# COMMAND ----------

chain_type_kwargs = {"prompt": prompt}
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 1}),
    chain_type_kwargs=chain_type_kwargs,
)
    

# COMMAND ----------

query = "What does Energex do?"
response = chain.run(query)

# COMMAND ----------

print(response)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ##Only answer from the Vector DB 

# COMMAND ----------

from langchain.chains.retrieval_qa.base import VectorDBQA
#from langchain.schema.vectorstore import VectorStore

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

question = 'What is the highest efficiency achieved for a solar cell?'
index.query(question=question, llm=llm)

# COMMAND ----------

question = "What is the population of the United States of America?"
index.query(question=question, llm=llm)

# COMMAND ----------


