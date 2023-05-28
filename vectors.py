import os
import openai
import pinecone

openai_api_key = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = ""
PINECONE_ENV = "us-west4-gcp-free"
PINECONE_TABLE_NAME = "my_test"
OPENAI_KEY = ""

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
openai.api_key = OPENAI_KEY



def create_pinecone_index(table_name, dimension=1536, metric="cosine", pod_type="p1"):
    if table_name not in pinecone.list_indexes():
        pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)

def complete(prompt):
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return res['choices'][0]['message']['content'].strip()

def get_ada_embedding(text):
    text = text.replace("\n", " ")
    print("hey")
    return openai.Embedding.create(input=text, model="text-embedding-ada-002")["data"][0]["embedding"]

def upsert_to_index(index, texts):
    pinecone_vectors = []
    for loopIndex, text in enumerate(texts, start=1):
        pinecone_vectors.append(("test-openai-"+str(loopIndex), get_ada_embedding(text), {"text": text}))
    index.upsert(vectors=pinecone_vectors)

def query_index(index, query_text, top_k=3):
    q_embedding = get_ada_embedding(query_text)
    pineQ = index.query(q_embedding, top_k=top_k, include_values=False, include_metadata=True)
    return pineQ

def print_results(pineQ):
    print(f"\033[36m" + str(pineQ) + "\033[0m")
    print("\n")
    for match in pineQ.matches:
        print(f"\033[1m\033[32m" + match.metadata['text'])

def main():
    create_pinecone_index(PINECONE_TABLE_NAME)
    index = pinecone.Index(PINECONE_TABLE_NAME)

    """ 
    texts = [
        "AI Agents as virtual employees are the future",
        "Vector Databases are the future",
        "AGI is not here....yet."
    ] 
    
    upsert_to_index(index, texts)
    """

    # query_text = "are vector dbs the future?"
    # results = query_index(index, query_text)
    # print_results(results)
    pinecone.deinit()

main()