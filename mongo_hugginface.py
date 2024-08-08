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

    def search_movies(self, query: str, genre_filter=None, year_filter=None, rated_filter=None,imdb_rating_filter=None):
        query_filter = {}
        if len(genre_filter)>0:
            query_filter['genres'] = {'$in': genre_filter}

        if year_filter:
            query_filter['year'] = {'$gte': year_filter[0], '$lte': year_filter[1]}

        if len(rated_filter)>0:
            query_filter['rated'] = {"$in":rated_filter}

        if imdb_rating_filter:
            query_filter['imdb.rating'] = {'$gte': imdb_rating_filter[0], '$lte': imdb_rating_filter[1]}

        
        #else:
        #    pass

        #query_filter = {
        #   "genres": {'$in': genre_filter},
        #    "rated": {"$in":rated_filter},
        #   "year":{'$gte': year_filter[0], '$lte': year_filter[1]}
        #}

        #st.write(query_filter)
        self.collection = self.db.movies
        initial_results = self.collection.find(query_filter, {'_id': 1})
        matching_ids = [doc['_id'] for doc in initial_results]

        query_vector = self.generate_embedding(query)
        pipeline = []

        pipeline.append({
            "$vectorSearch": {
                "queryVector": query_vector,
                "path": "plot_embedding_hf",
                "numCandidates": 5000,
                "limit": 10,
                "index": "PlotSemanticSearch",
            }
        })

        # Add a match stage to limit vector search to the initially filtered results
        pipeline.append({"$match": {"_id": {"$in": matching_ids}}})

        pipeline.append({"$project": {"score": {"$meta": "vectorSearchScore"}}})

        results = self.collection.aggregate(pipeline)

        return results