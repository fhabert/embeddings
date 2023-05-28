from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
from transformers import pipeline
import os
import json
import requests
from numpy.linalg import norm
import numpy as np

TOKEN = ""
URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/"
HEADERS = {"Accept": "Application/json", "Authorization": f"Bearer {TOKEN}"}

def openai_embedding():
    openai_api = os.getenv("OPENAI_API_KEY")
    print(openai_api)
    url_embedded = "https://api.openai.com/v1/embeddings"
    headers = {"Accept": "Application/json","Authorization": "Bearer "}
    payload = { "input": ["I want this to be embedded"],"model": "text-embedding-ada-002","options":{"wait_for_model":True}}
    web = requests.post(url_embedded, headers=headers, params=json.dumps(payload), auth=None)
    print("Status:", web.status_code)
    print(web.text)
    pass

def cosine_similarity(v1: np.array, v2: np.array) -> np.float64:
    num = np.dot(v1, v2)
    den = norm(v1)*norm(v2)
    return num/den

def total_cos_similarity(v: np.array) -> np.float64:
    l = len(v)
    sim_matrix = np.zeros((l, l))
    total_sim = 0
    for i in range(l):
        for j in range(l):
            sim = cosine_similarity(v[i][0][0], v[j][0][0])
            sim_matrix[i][j] = sim
            if i != j:
                total_sim += sim
    mean_sim = total_sim / (l*l-l)
    return mean_sim

def hf_embedding():
    model_id = "sentence-transformers/all-MiniLM-L6-v2" #direct embedings in english
    api_url = URL + model_id
    query = ["football"]
    response = requests.post(api_url, headers=HEADERS, json={"inputs": query, "options":{"wait_for_model":True}})
    return response.json()

def similarity():
    model_dif_languages = "xlm-mlm-enfr-1024"
    api_url = URL + model_dif_languages
    query = ["plante", "fleur", "flower"]
    response = requests.post(api_url, headers=HEADERS, json={"inputs": query, "options":{"wait_for_model":True}})
    formatted_output = np.array(response.json(), dtype=object)
    output = total_cos_similarity(np.array(formatted_output))
    return output

# call_openai()
embedded_output = similarity()
print(embedded_output)