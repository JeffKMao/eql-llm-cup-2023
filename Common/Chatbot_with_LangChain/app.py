import os
import requests
import json
import numpy as np
import pandas as pd
import re
import streamlit as st
from transformers import pipeline
from langchain.prompts import PromptTemplate



with st.sidebar:
    left_co, cent_co,last_co = st.columns(3)
    side_title = '<h1 style="color:#047fbf; font-size:24px;">Welcome to the ENERGEX LLM Experience</h1>'
    st.markdown(side_title, unsafe_allow_html=True) 
    st.markdown('', unsafe_allow_html=True)
    st.write('**About Buddy:**')
    st.markdown('*Buddy* is your virtual assistant who has its own knowledge base that sourced by Energex. Buddy can help you on things like outages, safety, managing your energy and using Energex Services.', unsafe_allow_html=True)
    st.markdown('', unsafe_allow_html=True)
    st.markdown('**To make the most of your conversations, please be specific and concise. Here are some examples:**', unsafe_allow_html=True)

    multi_rows = '''
    \n"How can I receive the instant notification about outages?"
    \n"How to safely connect a generator to my home?"
    \n"What is a demand tariff?"
    \n"Do I need to change my meter?"
    '''
    st.markdown(multi_rows)


def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}
    
def score_model(dataset):
    url = 'https://dbc-eb788f31-6c73.cloud.databricks.com/serving-endpoints/Energex-L2M1/invocations'
    databricks_token = '<Type Your Personal Token Here>'
    headers = {'Authorization': f'Bearer {databricks_token}', 'Content-Type': 'application/json'}
    # headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello 👋\n\nMy name is Buddy. How can I help?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def call_llm(prompt):
    string_dialogue = """You are a helpful, respectful and honest assistant. Your name is Buddy. You have your own knowledge base that sourced by Energex. You respond as "Assistant". You do not respond as 'User' or pretend to be 'User'. Always answer as helpfully as possible, while being safe. Answer the question from {text_input}. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use five sentences maximum. Keep the answer as concise as possible. """
    

    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "You: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Buddy: " + dict_message["content"] + "\n\n"

    prompt_template = PromptTemplate(input_variables=['text_input'], 
                                    template=string_dialogue)
    prompt_query = prompt_template.format(text_input='prompt')

    # Energex-L2M1
    input_prompt = pd.DataFrame({"prompt": [prompt_query],
                                 "temperature": 0.0, 
                                 "max_tokens": 512})

    response = score_model(input_prompt)
    return response

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = call_llm(prompt)
            placeholder = st.empty()
            full_response = ''
            for i, k in response['predictions'][0].items():
                text_output = re.sub(r"<@@@>.*", "", str(k))
                full_response += text_output
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)