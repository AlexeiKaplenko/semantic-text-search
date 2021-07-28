import streamlit as st
from nltk import tokenize
import asyncio
import aiohttp
import json
import requests
import ast
from unsync import unsync
import time

# @unsync
def analyze_sequence(sequence, query):

    # url = "https://d88oumkjii.execute-api.eu-central-1.amazonaws.com/zero-shot-classification-2-pytorchEndpoint-AUCRYD6WXUK3"

    # url = "https://zeu90cba02.execute-api.eu-central-1.amazonaws.com/zero-shot-classification-3-pytorchEndpoint-B0UPVS3RAF1C"

    url = "https://pprvz8rt3h.execute-api.eu-central-1.amazonaws.com/zero-shot-classification-6"

    headers = {"content-type": "application/json"}

    data_dict = {}
    data_dict["label"] = query

    data_dict["premise"] = sequence
    data = json.dumps(data_dict)
    response = requests.post(url, data=data, headers=headers)
    
    return response.text

def main():

    st.title('semantic search engine')

    st.subheader("Text you'd like to analyze")
    text = st.text_input('Enter text') #text is stored in this variable

    st.subheader("Query you'd like to find")
    query = st.text_input('Enter query') #text is stored in this variable

    confidence_threshold = st.sidebar.slider("Confidence threshold", min_value=0., max_value=1., value=0.8, step=0.01)

    sequences=tokenize.sent_tokenize(text)

    if st.button("Search"):

        if text is not None and query is not None:

            # output = []

            # for sequence in sequences:
            #     sequence_output = analyze_sequence(sequence, query)
            #     output.append(sequence_output)

            st.write("Relevant sentences:")

            try:
                for sequence in sequences:
                    # out_dict = ast.literal_eval(output[i].result())
                    sequence_output = analyze_sequence(sequence, query)


                    out_dict = ast.literal_eval(sequence_output)

                    if out_dict['probability'] != None:

                        if float(out_dict['probability']) >= confidence_threshold:
                            probability = type(out_dict['probability'])

                            st.write(sequence, '(confidence:', out_dict['probability'][:4],')')    

                    else:
                        st.write('probability is None')
                        continue   
            except:
                st.write('Neural network is initializing, please repeat search in 1 minute')

        else:
            st.write('Please fill text and/or query')

if __name__ == "__main__":
    main()




