# Singular Value Decomposition Lab

## Task 1: SVD Implementation

Implementation of Singular Value Decomposition (SVD) algorithm from scratch.

### Algorithm

1. Compute A^T · A
2. Find eigenvalues and eigenvectors
3. Calculate singular values: σᵢ = √λᵢ
4. Form matrices U, Σ, V^T
5. Verify reconstruction: A = U · Σ · V^T

### Код

```python
import numpy as np

def svd_implementation(matrix):
    matrix_transpose = matrix.T
    first_symmetric_matrix = matrix_transpose @ matrix
    eigenvalues, eigenvectors = np.linalg.eig(first_symmetric_matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    singular_values = np.sqrt(eigenvalues)
    sorted_indx = np.argsort(singular_values)[::-1]
    singular_values = singular_values[sorted_indx]
    V = eigenvectors[:,sorted_indx]
    sigma = np.diag(singular_values)
    U = matrix @ V
    for i in range (len(singular_values)):
        if not singular_values[i] == 0:
            U[:, i] = U[:, i] / singular_values[i]
        else:
            print("There is division by zero")

    matrix_reconstructed = U@sigma@V.T
    is_true = np.isclose(matrix_reconstructed, matrix)
    if np.all(is_true):
        print("True")
    else:
        print("False")

# Test
test_matrix = np.array([[2, 3, 4], [1, 9, 4], [4, 3, 2]])
svd_implementation(test_matrix)
```

### Result

Function outputs `True` if the reconstructed matrix matches the original.

---

## Task 2: Movie Recommendation System (Part 1)

Building a collaborative filtering recommendation system using SVD on movie ratings data.

### Data Preprocessing

1. Load ratings dataset (userId, movieId, rating)
2. Create user-movie rating matrix
3. Filter: keep users with 200+ ratings, movies with 100+ ratings
4. Fill missing values with 2.5 (neutral rating)
5. Demean ratings by user average

### SVD Decomposition

Apply SVD with k=3 factors to extract:
- **U matrix**: User preferences in latent space
- **Σ (sigma)**: Importance weights of each factor
- **V^T matrix**: Movie characteristics in latent space

### Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

file_path = 'datasets/ratings.csv'
df = pd.read_csv(file_path)
ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

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
```

Creates two 3D scatter plots:
- **User space**

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/6b178dbb-a2ab-4e9e-bf84-cda0540b3508" />

First 20 users plotted in 3-factor latent space

- **Movie space**

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/d0d33732-10ee-4f7a-9b05-5688399c51f0" />

All movies plotted in 3-factor latent space


---

## Task 3: Generating Movie Recommendations

Complete recommendation system that predicts ratings for unseen movies and generates personalized top-10 recommendations.

### Algorithm

1. Load and preprocess ratings data (keep users with 33+ ratings, movies with 44+ ratings)
2. Fill missing values with 2.5 and demean by user average
3. Apply SVD with k=3 to decompose the matrix: R = U · Σ · V^T
4. Reconstruct full rating matrix with predictions
5. Extract only predicted ratings (originally missing values)
6. Sort predictions and recommend top-10 highest-rated movies

### Code

```python
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
```

### Example Output

**Top 10 recommendations for user with id 5:**
- Die Hard: With a Vengeance (1995) — Action|Crime|Thriller
- Star Wars: Episode IV - A New Hope (1977) — Action|Adventure|Sci-Fi
- Forrest Gump (1994) — Comedy|Drama|Romance|War
- Speed (1994) — Action|Romance|Thriller
- Jurassic Park (1993) — Action|Adventure|Sci-Fi|Thriller
- Silence of the Lambs, The (1991) — Crime|Horror|Thriller
- Mission: Impossible (1996) — Action|Adventure|Mystery|Thriller
- Independence Day (a.k.a. ID4) (1996) — Action|Adventure|Sci-Fi|Thriller
- Star Wars: Episode V - The Empire Strikes Back (1980) — Action|Adventure|Sci-Fi
- Star Wars: Episode VI - Return of the Jedi (1983) — Action|Adventure|Sci-Fi
