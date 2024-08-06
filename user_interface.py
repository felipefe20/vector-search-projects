import streamlit as st
from mongo_hugginface import EmbeddingsUtil

# Initialize the embeddings utility
embeddings_util = EmbeddingsUtil()

# Streamlit UI
st.title('Movie Recommender')

# Example filter UI elements
genre_filter = st.multiselect('Select genres:', options=['Action', 'Adventure', 'Animation', 'Biography', 
                                                         'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                                                           'Fantasy', 'Film-Noir', 'History', 'Horror', 
                                                           'Music', 'Musical', 'Mystery', 'News', 'Romance',
                                                             'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 
                                                             'Thriller', 'War', 'Western'])
year_filter = st.slider('Select year range:', min_value=1900, max_value=2023, value=(1990, 2023))
rated_filter = st.multiselect('Select rating:', options=["Any",'AO', 'APPROVED', 'Approved', 'G', 
                                                         'GP', 'M', 'Not Rated', 'OPEN', 'PASSED', 
                                                         'PG', 'PG-13', 'R', 'TV-14', 'TV-G', 'TV-MA',
                                                           'TV-PG', 'TV-Y7'])

#st.write("genre_filter", genre_filter)
#st.write("year_filter", year_filter)
#st.write("rated_filter", rated_filter)


# Chat input
user_input = st.text_input("Ask for movie recommendations:")

if user_input:
    # Search for movies based on the input query

    results = embeddings_util.search_movies(user_input, genre_filter, year_filter, rated_filter)
    #st.write("results", results)
    # Display results
    if results:
        for document in results:
            #st.write(i)
            #document=results.find_one({"_id":i["_id"]})
            #st.write(document)
            try: 
                st.write(f"**Movie Name:** {document['title']}")
                st.image(document['poster'], width=200)
                st.write(f"**Movie Plot:** {document['plot']}")
                st.write(f"**Movie Rating:** {document['rated']}")
                st.write(f"**Movie Genres:** {document['genres']}")
                st.write(f"**Movie Year:** {document['year']}")
                #st.write(f"Similarity score: {i['score']}")
                st.write("-----------------------------------")

                #st.write(document)
            except:
                pass
    else:
        st.write("No movies found. Try another query!")


