import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

file_path = 'datasets/ratings.csv'
df = pd.read_csv(file_path)
ratings_matrix = df.pivot(index='userId', columns='movieId',
values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=200, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=100, axis=1)
nparr = df.values
ratings_matrix_filled = ratings_matrix.fillna(2.5)
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)
U, sigma, Vt = svds(R_demeaned, k=3)

print(f" User preferences:\n{U},\n\n Films characteristics:\n{Vt}\n\n Coef:\n{sigma}")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

user_factor1 = U[:, 0]
user_factor2 = U[:, 1]
user_factor3 = U[:, 2]
ax.scatter(user_factor1[:20], user_factor2[:20], user_factor3[:20])
ax.set_xlabel("Factor 1")
ax.set_ylabel("Factor 2")
ax.set_zlabel("Factor 3")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

film_factor1 = Vt[0, :]
film_factor2 = Vt[1, :]
film_factor3 = Vt[2, :]

ax.scatter(film_factor1, film_factor2, film_factor3)

ax.set_xlabel("Factor 1")
ax.set_ylabel("Factor 2")
ax.set_zlabel("Factor 3")

plt.show()