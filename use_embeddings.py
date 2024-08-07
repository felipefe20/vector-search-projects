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


db = client.sample_mflix
collection = db.movies

query = "Dinosaurs"

#query_filter = {"year":{'$gte': 1975, '$lte': 1977}}
query_filter={
  "genres": {
    "$in": ["Sci-Fi"]
  },
  #"rated": {
  #  "$in": [
  #    "Any"
  #  ]
  #},
  "year": {
    "$gte": 1900,
    "$lte": 1980
  }
}
initial_results = collection.find(query_filter, {'_id': 1})
matching_ids = [doc['_id'] for doc in initial_results]

query_vector = generate_embedding(query)
pipeline = []

pipeline.append({
    "$vectorSearch": {
        "queryVector": query_vector,
        "path": "plot_embedding_hf",
        "numCandidates": 100,
        "limit": 10,
        "index": "PlotSemanticSearch",
    }
})

# Add a match stage to limit vector search to the initially filtered results
pipeline.append({"$match": {"_id": {"$in": matching_ids}}})

pipeline.append({"$project": {"score": {"$meta": "vectorSearchScore"}}})

results = collection.aggregate(pipeline)

for i in results:
    document=collection.find_one({"_id":i["_id"]})
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["fullplot"]}\n')
    print(f'Movie Genre: {document["genres"]}')
    print(f'Movie Year: {document["year"]}')
    print("score:", i["score"], "\n")