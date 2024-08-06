
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os

import requests 

# Load the environment variables
load_dotenv()

# Get the MongoDB URI from the environment
uri = os.getenv('MONGODB_URI')

# Create a new client and connect to the server
client = MongoClient(uri)


hf_token=os.getenv('HF_token')
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

#Generate embedding function
def generate_embedding(text: str) -> list[float]:

  response = requests.post(
    embedding_url,
    headers={"Authorization": f"Bearer {hf_token}"},
    json={"inputs": text})

  if response.status_code != 200:
    raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

  return response.json()

query = "Felipe  is the best data engineer in the world!"
#print(generate_embedding(query))



db = client.sample_mflix
collection = db.movies

for doc in collection.find({'plot':{"$exists": True}}).limit(50):
   doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
   collection.replace_one({'_id': doc['_id']}, doc)
