import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_df = pd.read_csv('mflix.movies.csv')
print(movies_df.head())
print(movies_df.info())

# Data Handling
# Getting the summary of missing values in each column

missing_values = movies_df.isnull().sum()
missing_values = missing_values[missing_values > 0]
missing_values

# Dropping columns with excessive missing values
columns_to_drop = ['directors[1]', 'writers[1]', 'genres[2]']
movies_df.drop(columns=columns_to_drop, inplace=True)

#Merging genres and Cast into a single column
movies_df['Genre'] = movies_df[['genres[0]', 'genres[1]']].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)
movies_df['Cast'] = movies_df[['cast[0]', 'cast[1]', 'cast[2]', 'cast[3]']].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)

# Dropping the columns that were merged
movies_df.drop(columns=['genres[0]', 'genres[1]', 'cast[0]', 'cast[1]', 'cast[2]', 'cast[3]'], inplace=True)

#Removing rows with null values in specific columns because we can't fill the missing values with mode or media, since these are too unique.
columns_to_check = ['poster', 'fullplot', 'languages[0]', 'released', 'directors[0]','writers[0]', 'countries[0]','poster', 'awards.text', 'lastupdated', 'imdb.votes', 'type', 'rated']
movies_df.dropna(subset=columns_to_check, inplace=True)

numerical_columns = ['runtime', 'imdb.rating', 'imdb.votes', 'tomatoes.viewer.rating']
for column in numerical_columns:
    movies_df[column].fillna(movies_df[column].median(), inplace=True)
    
#Filling missing values in 'rated' with a placeholder
movies_df['rated'].fillna('Not Rated', inplace=True)

#Checking for missing values
missing_values = movies_df.isnull().sum()
missing_values = missing_values[missing_values > 0]
missing_values

# Renaming the columns with understandable names.

movies_df.rename(columns={
    'plot': 'Plot',
    'runtime': 'Runtime',
    'num_mflix_comments': 'Comments',
    'title': 'Title',
    'fullplot': 'FullPlot',
    'languages[0]': 'Language',
    'released': 'Released',
    'directors[0]': 'Director',
    'writers[0]': 'Writer',
    'awards.wins': 'Awards',
    'awards.nominations': 'Nominations',
    'year': 'Year',
    'imdb.rating': 'IMDB Rating',
    'countries[0]': 'Countries',
    'tomatoes.viewer.rating': 'Tomatoes Rating'
}, inplace=True)


#Now checking the dataframe information
# Display the updated dataset information and first few rows
updated_info = movies_df.info()
updated_head = movies_df.head()
updated_info, updated_head

# Model 1
# Vectorize the Genre column using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['Genre'])

#Calculate the cosine similarity between the genre vectors
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#Function to get movie recommendations based on genre similarity
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = movies_df[movies_df['Title'].str.lower() == title.lower()].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the 10 most similar movies
    sim_scores = sim_scores[1:11]  # Exclude the first movie (itself)

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies_df['Title'].iloc[movie_indices]


#Example usage: Get recommendations for a specific movie
recommendations = get_recommendations("A Walk in the Sun")
recommendations

# Model 2
#Function to get movie recommendations based on genre and released year as input

#Combine Genre and Released Year into a new feature
movies_df['Genre_Released'] = movies_df['Genre'] + ' ' + movies_df['Released'].astype(str)

#Vectorize the combined Genre_Released column using TF-IDF
tfidf_vectorizer_combined = TfidfVectorizer(max_features=500)
tfidf_matrix_combined = tfidf_vectorizer_combined.fit_transform(movies_df['Genre_Released'])

def recommend_movies_by_genre_year(genre, year, tfidf_matrix=tfidf_matrix_combined, n_recommendations=10):
    # Combine genre and year into a single string
    input_combined = genre + ' ' + str(year)
    
    # Transform the input into a TF-IDF vector
    input_vector = tfidf_vectorizer_combined.transform([input_combined])
    
    # Calculate the cosine similarity of the input with all movies
    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix_combined).flatten()
    
    # Get the indices of the most similar movies
    sim_indices = cosine_similarities.argsort()[-(n_recommendations + 1):][::-1][1:]

    # Return the top n most similar movies
    return movies_df['Title'].iloc[sim_indices]

#Example usage: Get recommendations for a specific genre and release year
recommended_movies = recommend_movies_by_genre_year("Action", 2012)
recommended_movies

