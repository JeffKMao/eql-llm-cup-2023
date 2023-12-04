
import streamlit as st
#from PIL import Image
import streamlit.components.v1 as components
import os
import requests
import numpy as np
import pandas as pd
import json
import re
from better_profanity import profanity

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    #url = 'https://dbc-eb788f31-6c73.cloud.databricks.com/serving-endpoints/energex-dolly-7b/invocations'
    #url = 'https://dbc-eb788f31-6c73.cloud.databricks.com/serving-endpoints/Energex-L2M1-prod/invocations'
    url = 'https://dbc-eb788f31-6c73.cloud.databricks.com/serving-endpoints/Energex-L2M1/invocations'

    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

def change_html_page(label):
    # Source Website is blocking embedding iFrame. To demonstrate capability, substituting the direct iFrame with local htm files 
    label_dict = {
        'https://www.energex.com.au/outages/be-prepared-for-outages': 'prepare-outages.htm', 
        'https://www.energex.com.au/outages/outage-finder': 'outage-finder.htm', 
        'https://www.energex.com.au/manage-your-energy/smarter-energy/electric-vehicles-ev': 'ev.htm',
        'https://www.energex.com.au/outages': 'outages.htm', 
        'https://www.energex.com.au/our-network/network-pricing-and-tariffs': 'tariffs.htm',
        'https://www.energex.com.au/our-services/metering': 'metering.htm',
        'https://www.energex.com.au/manage-your-energy': 'manage-your-energy.htm',
        'https://www.energex.com.au/safety/kids-safety': 'kids-safety.htm',
        'https://www.energex.com.au/outages/hot-water-issues-cold-water': 'hot-water.htm',
        'https://www.energex.com.au/safety/safety-at-home-or-work': 'safety-work-home.htm',
        'https://www.energex.com.au/safety': 'safety.htm',
        'https://www.energex.com.au/manage-your-energy/save-money-and-electricity': 'save-money.htm',
        'https://www.energex.com.au/our-services/self-service': 'self-service.htm', 
        'https://www.energex.com.au/manage-your-energy/smarter-energy/solar-pv': 'solar.htm', 
        'https://www.energex.com.au/outages/storms-and-disasters': 'storms.htm', 
        'https://www.energex.com.au/safety/working-near-powerlines': 'powerlines.htm'
        }

    if label in label_dict.keys():
        print('Found: ' + str(label))
        st.session_state.url = label_dict[label]
        print('Change label to: ' + str(label_dict[label]))
        print('State is: ' + st.session_state.url)
    else:
        st.session_state.url = 'energex.htm'

def button_click():
    if st.session_state["text"] != '':
        if profanity.contains_profanity(st.session_state["text"]):
            st.session_state.text = ''
            st.session_state.output_text = 'Sorry - that is inappropriate.\n\n' + st.session_state.output_text 
        else:
            prompt = pd.DataFrame({"prompt":st.session_state["text"], "temperature": [0.5], "max_tokens": [2048]})
            api_response = score_model(prompt)
            response = re.search('(.*)<@@@>', str(api_response['predictions'][0]['0'])).group(1)
            hyperlink = re.search('<@@@>(.*)', str(api_response['predictions'][0]['0'])).group(1)
            
            #output.write(str(response))
            #st.session_state.output_text = response + '\n\n' + '> ' + st.session_state.prev_text + '\n' + st.session_state.output_text 
            st.session_state.output_text = response + '\n\n\>'  + st.session_state.prev_text +  '\n\n' + st.session_state.output_text 
            st.session_state.prev_text = st.session_state["text"]
            output.write(str(hyperlink))
            change_html_page(hyperlink)
            # output.write(str(api_response['predictions'][0]['0']))
            # output.write(str(api_response))
            # output.write('Response: ' + str(response))
            #output.write('Classification Link: ' + str(hyperlink))

st.set_page_config(page_title="ENERGEX LLM Experience", layout="wide")
#st.title('Welcome to the ENERGEX LLM Experience')
st.title('Welcome to the ENERGEX LLM')

if 'url' not in st.session_state:
    st.session_state.url = 'energex.htm'
if 'text' not in st.session_state:
    st.session_state.text = ''
if 'prev_text' not in st.session_state:
    st.session_state.prev_text = ''
if 'output_text' not in st.session_state:
    st.session_state.output_text = ''
    print('Resetting output_text')


webpage = st.empty()
with webpage.container():
    # st.write(st.session_state.url)
    p = open('/Volumes/eql_llm/app_serving/eql_app_serving/website/' + st.session_state.url )
    components.html(p.read(),width=1200, height=900, scrolling=True)
    #components.html(p.read(),width=600, height=400, scrolling=True)

with st.sidebar:
    form = st.form("my_form")
    form.text_input("What would you like to know?", key="text")
    form.form_submit_button("Submit", on_click=button_click)

    output = st.container()
    output.write(st.session_state.output_text)

    # output = st.chat_message("msg")
    # output.write(st.session_state.output_text)


