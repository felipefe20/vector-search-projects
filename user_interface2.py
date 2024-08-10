from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
import streamlit as st
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


# Streamlit UI
st.title('Movie Recommender')

import streamlit as st

# Create columns
col1, col2, col3, col4 = st.columns(4)

# Place each filter in a separate column
with col1:
    genre_filter = st.multiselect('Select genres:', options=[
        'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
        'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
        'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western'
    ])

with col2:
    year_filter = st.slider('Select year range:', min_value=1900, max_value=2023, value=(1900, 2023))

with col3:
    rated_filter = st.multiselect('Select rating:', options=[
        'AO', 'APPROVED', 'Approved', 'G', 'GP', 'M', 'Not Rated', 'OPEN', 'PASSED', 
        'PG', 'PG-13', 'R', 'TV-14', 'TV-G', 'TV-MA', 'TV-PG', 'TV-Y7'
    ])

with col4:
    imdb_rating_filter = st.slider('Select IMDB rating:', min_value=0.0, max_value=10.0, value=(0.0, 10.0), step=0.1)


year_filter=list(year_filter)



#st.write(genre_filter)
#st.write(year_filter)
#st.write(rated_filter)

# Chat input
col1, col2 = st.columns(2)
with col1:
  query= st.text_input("Ask for movie recommendations")
with col2:
  #How many movies to recommend

  num_movies = st.slider('Number of movies to recommend:', min_value=1, max_value=50, value=1)


#query = "Outer space adventure"
if query:
    # Search for movies based on the input query

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

  initial_results = collection.find(query_filter, {'_id': 1})
  matching_ids = [doc['_id'] for doc in initial_results]

  query_vector = generate_embedding(query)
  pipeline = []

  pipeline.append({
      "$vectorSearch": {
          "queryVector": query_vector,
          "path": "plot_embedding_hf",
          "numCandidates": 5000,
          "limit": num_movies,
          "index": "PlotSemanticSearch",
      }
  })

  # Add a match stage to limit vector search to the initially filtered results
  pipeline.append({"$match": {"_id": {"$in": matching_ids}}})

  pipeline.append({"$project": {"score": {"$meta": "vectorSearchScore"}}})

  results = collection.aggregate(pipeline)
  counter=1
  while counter<= num_movies:
    for i in results:
        document=collection.find_one({"_id":i["_id"]})
        col1,col2, col3= st.columns(3)
        with col1:
          st.write(f"## {counter}")
          counter=counter+1
          st.write("Score:", i.get("score", "N/A"), "\n")
          st.write(f'*Title*: {document.get("title", "N/A")}')
          st.write("*imdb rating*:", document.get("imdb", {}).get("rating", "N/A"), "\n")
          st.write(f'*Genre*: {document.get("genres", "N/A")}')
          st.write(f'*Year*: {document.get("year", "N/A")}')
          st.write(f'*rated*: {document.get("rated", "N/A")}')
        with col2:
          st.write(f'*Plot*: {document.get("fullplot", "N/A")}')
        with col3:
          try:
            st.image(document.get("poster"), width=200)
          except:
            pass
        st.write("--------------------------------------------------------------------")

else: 
  st.write("No movies found. Try another query!")