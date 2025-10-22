import pandas as pd
from Levenshtein import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, init
init(autoreset=True)

dataset_path        = r"Datasets/latest-small/"
movies              = pd.read_csv(dataset_path+r"movies.csv")

movies['genres']    = movies['genres'].fillna('').apply(lambda x: x.replace('|', ' '))
moviesTitle         = movies['title'].fillna('').to_list()

tfidf               = TfidfVectorizer(stop_words='english')
tfidf_matrix        = tfidf.fit_transform(movies['genres'])

cosineSim           = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices             = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def recommend_me(title, cosineSim=cosineSim):
    index           = indices[title]
    simScores       = list(enumerate(cosineSim[index]))

    simScores       = sorted(simScores, key=lambda x: x[1], reverse=True)
    simScores       = simScores[1:11] # Fetching the top 10

    movieIndices    = [i[0] for i in simScores]

    return movies['title'].iloc[movieIndices]

while True:
    user_input          = input("Enter a movie: ").strip()

    if user_input in moviesTitle:
        print(Fore.GREEN+f"Here are the top 10 movies similar to {user_input}:\n", recommend_me(user_input), '\n')
    else:
        dist_all        = {title: distance(user_input, title) for title in moviesTitle}

        closest_match   = min(dist_all, key=dist_all.get)
        closest_dist    = dist_all[closest_match]

        print(Fore.RED+f"I apologise, '{user_input}' was not found in the database.")
        print(Fore.YELLOW+f"Did you mean: '{closest_match}'?\n")
