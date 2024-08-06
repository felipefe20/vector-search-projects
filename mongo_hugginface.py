from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
import requests


# Load the environment variables
load_dotenv()

class EmbeddingsUtil:
    def __init__(self):
        self.uri = os.getenv('MONGODB_URI')
        self.client = MongoClient(self.uri)
        self.db = self.client.sample_mflix
        self.collection = self.db.movies
        self.hf_token = os.getenv('HF_token')
        self.embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

    def generate_embedding(self, text: str) -> list:
        response = requests.post(
            self.embedding_url,
            headers={"Authorization": f"Bearer {self.hf_token}"},
            json={"inputs": text}
        )

        if response.status_code != 200:
            raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

        return response.json()

    def search_movies(self, query: str, genres=None, year_range=None, rated=None):
        query_filter = {}
        if len(genres)>0:
            query_filter['genres'] = {'$in': genres}
        
        if year_range:
            query_filter['year'] = {'$gte': year_range[0], '$lte': year_range[1]}
        
        if "Any" not in rated:
            query_filter['rated'] = {"$in":rated}
        else:
            pass
        
        query_vector = self.generate_embedding(query)
        
        # Start the aggregation pipeline with a $match stage if there are filters to apply
        pipeline = []
        
        
        # Add the $vectorSearch stage to the pipeline
        pipeline.append({
            "$vectorSearch": {
                "queryVector": query_vector,
                "path": "plot_embedding_hf",
                "numCandidates": 100,
                "limit": 4,
                "index": "PlotSemanticSearch",
            }
        })
        
        #pipeline.append({"$project": 
      
        #  {"score": {"$meta": "vectorSearchScore"}}

        #                })
        if query_filter:
            pipeline.append({"$match": query_filter})
        # Use the aggregate method directly on the collection with the constructed pipeline
        results = self.collection.aggregate(pipeline)
        return results