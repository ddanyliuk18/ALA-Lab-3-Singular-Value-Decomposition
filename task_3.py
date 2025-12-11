import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

def prepare_recommendation_data(k=3):
    file_path = 'datasets/ratings.csv'
    df = pd.read_csv(file_path)
    ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')
    ratings_matrix = ratings_matrix.dropna(thresh=33, axis=0)
    ratings_matrix = ratings_matrix.dropna(thresh=44, axis=1)

    ratings_matrix_filled = ratings_matrix.fillna(2.5)
    R = ratings_matrix_filled.values
    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)

    U, sigma, Vt = svds(R_demeaned, k=k)
    Sigma = np.diag(sigma)

    recomposed = U @ Sigma @ Vt
    final_prognosed_ratings = recomposed + user_ratings_mean.reshape(-1, 1)

    only_predictions = pd.DataFrame(index=ratings_matrix.index,
                                    columns=ratings_matrix.columns)
    for i in range(len(ratings_matrix.index)):
        for j in range(len(ratings_matrix.columns)):
            if pd.isna(ratings_matrix.iloc[i, j]):
                only_predictions.iloc[i, j] = final_prognosed_ratings[i, j]
            else:
                only_predictions.iloc[i, j] = np.nan
    print(f"Real ratings: {user_ratings_mean}\n\nPrognosed: {final_prognosed_ratings}")
    return only_predictions

def recommendations(user_id, preds):
    movies_data = pd.read_csv("datasets/movies.csv")
    preds = preds.loc[user_id]
    preds = preds.dropna()
    sorted_indx = np.argsort(preds.values)[::-1]
    top_10_ids = preds.index[sorted_indx][:10]
    recommended = movies_data[movies_data["movieId"].isin(top_10_ids)]
    print(f"Top 10 recommendations for user with id {user_id}:\n")
    for _, row in recommended.iterrows():
        print(f"{row['title']} — {row['genres']}")

predictions = prepare_recommendation_data(3)
recommendations(5, predictions)