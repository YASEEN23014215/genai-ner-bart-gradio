## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM :
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT :
Build an interactive Named Entity Recognition (NER) system that identifies and highlights entities such as persons, organizations, locations, and miscellaneous items in text. The system should use the pre-trained **dslim/bert-base-NER** model via Hugging Face, merge multi-token entities, and provide real-time feedback for user-input text, handling API responses gracefully.

### DESIGN STEPS :

#### STEP 1 :
Input and API Call: Take user input text and send it to the Hugging Face NER API using the pre-trained dslim/bert-base-NER model.

#### STEP 2 :
Process and Merge Tokens: Receive the list of predicted tokens and merge subword tokens that belong to the same entity to form complete named entities.

#### STEP 3 :
Output and Display: Return the original text along with the recognized entities and their labels, highlighting them interactively in the Gradio interface.


### PROGRAM :
```
import os
import requests
import json
import gradio as gr
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
hf_api_key = os.environ.get("HF_API_KEY")
API_URL = os.environ.get(
    "HF_API_NER_BASE",
    "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
)

def get_completion(inputs, parameters=None, ENDPOINT_URL=API_URL):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters is not None:
        data["parameters"] = parameters

    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(data))
    
    try:
        return response.json()
    except json.JSONDecodeError:
        print(response.text)
        return []


def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if not all(k in token for k in ("start", "end", "entity", "word")):
            continue
        if merged_tokens and token["entity"].startswith("I-") and \
           merged_tokens[-1]["entity"].endswith(token["entity"][2:]):
            last = merged_tokens[-1]
            last["word"] += token["word"].replace("##", "")
            last["end"] = token["end"]
            last["score"] = (last["score"] + token["score"]) / 2
        else:
            merged_tokens.append(token)
    return merged_tokens

def ner(input_text):
    output = get_completion(input_text)
    if not output or not isinstance(output, list):
        return {"text": input_text, "entities": []}
    merged = merge_tokens(output)
    return {"text": input_text, "entities": merged}

gr.close_all()

```
### OUTPUT :
<img width="1192" height="761" alt="image" src="https://github.com/user-attachments/assets/7de0d867-60c4-4b96-9ca5-1e3a883f92fd" />


### RESULT :
Thus, the Python program to design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation was executed successfully.
