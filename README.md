# ALA-Lab-3-Singular-Value-Decomposition

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

``` Output
User preferences:
[[-0.11701519 -0.0486775   0.10423191]
 [ 0.11847179 -0.08575014  0.25393107]
 [ 0.04659884 -0.2114704  -0.06167357]
 [-0.03369839 -0.03431137 -0.01493453]
 [-0.11315672  0.06023676  0.09295234]
 [ 0.          0.          0.        ]
 [-0.00176799 -0.06557723  0.04769789]
 [ 0.0223581   0.03028765 -0.01808238]
 [ 0.14094734  0.18267287 -0.0629276 ]
 [-0.02632272 -0.00939526 -0.06325497]
 [ 0.          0.          0.        ]
 [ 0.0808657   0.0546264  -0.07019834]
 [ 0.05670146  0.0201186   0.16508456]
 [-0.12109341 -0.04722163  0.04750486]
 [-0.02915682 -0.03123462 -0.02748431]
 [ 0.09201204  0.09678061  0.06990315]
 [ 0.01555938  0.02934068 -0.06364908]
 [-0.01879314 -0.00076514 -0.16784247]
 [-0.13257457 -0.07801818  0.16788099]
 [-0.00475961 -0.0016749  -0.01720317]
 [-0.05804253 -0.12877441  0.02109279]
 [ 0.15642742 -0.0874789   0.00269698]
 [-0.05516555  0.1449462   0.06380476]
 [-0.07978081  0.11678833  0.02377067]
 [ 0.05583846  0.06600087 -0.03290081]
 [ 0.04613378 -0.10703474 -0.0306504 ]
 [ 0.02480749  0.09899566 -0.10530416]
 [ 0.2407498  -0.20119048 -0.03362347]
 [-0.02039617 -0.0028409  -0.13670009]
 [ 0.          0.          0.        ]
 [-0.06997642 -0.04425275 -0.17210174]
 [-0.0701431   0.00121406 -0.19379991]
 [-0.12727059  0.11871351  0.02478733]
 [-0.00861979 -0.00162088 -0.04417727]
 [ 0.06835485  0.11517892 -0.16950573]
 [ 0.01756951  0.1302447   0.05524352]
 [-0.1166273  -0.12493847 -0.10993725]
 [ 0.13983835 -0.0254705  -0.15333746]
 [-0.01789371  0.00096379 -0.1776134 ]
 [-0.01847383  0.00638696  0.18189402]
 [-0.03280745  0.17523385  0.04572238]
 [ 0.          0.          0.        ]
 [-0.04357995  0.1549044  -0.00220812]
 [-0.02758278  0.0724731   0.03190238]
 [ 0.04982728 -0.06162196 -0.02537447]
 [ 0.03342899 -0.00835174 -0.02287322]
 [-0.04510409  0.01568567  0.25033393]
 [-0.11117069  0.12441214  0.07500741]
 [ 0.00454157  0.00307675 -0.01254978]
 [ 0.1823215   0.07396322 -0.11078117]
 [-0.02039617 -0.0028409  -0.13670009]
 [ 0.03119594  0.03050668  0.05584784]
 [ 0.00402514  0.06344745  0.01041859]
 [-0.18074746  0.0297445   0.17016787]
 [ 0.          0.          0.        ]
 [-0.08609037  0.04813435  0.08401834]
 [-0.05396602 -0.021262    0.03956617]
 [-0.05899504 -0.06560001 -0.01544474]
 [ 0.05556905  0.02333776 -0.07070855]
 [-0.08850504  0.13600582 -0.07476986]
 [-0.07822237 -0.02404887 -0.12410802]
 [ 0.16975374 -0.01927979  0.01776896]
 [ 0.16849197  0.17155686  0.07467264]
 [-0.03092482 -0.09681185  0.02021357]
 [ 0.16594664 -0.08412909  0.03710332]
 [-0.00976624  0.099684    0.02636978]
 [ 0.08589643 -0.01000386  0.02306624]
 [-0.01379139  0.03623655  0.01595119]
 [ 0.05646848  0.02506669 -0.08047948]
 [ 0.03241956 -0.09897288  0.16844678]
 [-0.08299047  0.18209251 -0.07088334]
 [ 0.05641712 -0.01899827 -0.08853427]
 [ 0.11774117  0.19472125 -0.0270195 ]
 [ 0.21160687  0.12108811  0.10790393]
 [ 0.07460921 -0.07893091  0.03635779]
 [-0.07526162 -0.02583183 -0.08736298]
 [ 0.04137417 -0.10870965 -0.04785357]
 [ 0.04297548  0.00233439 -0.11724962]
 [ 0.03614949 -0.0059489  -0.03403357]
 [-0.08747047 -0.09370385 -0.08245294]
 [ 0.14399203  0.05386722  0.04828192]
 [ 0.07966721 -0.13622485  0.00083963]
 [ 0.03155484  0.05587767 -0.06779225]
 [ 0.09217872  0.0513138   0.09160132]
 [ 0.04294821 -0.00500193  0.01153312]
 [-0.15976946 -0.08180605 -0.01438579]
 [-0.00924982  0.0393133   0.00340141]
 [ 0.142006   -0.01030816  0.06622685]
 [ 0.0560565   0.06459902 -0.00314785]
 [ 0.00908314  0.00615351 -0.02509957]
 [ 0.          0.          0.        ]
 [ 0.05693355 -0.07936897 -0.11150265]
 [-0.0120234   0.10181378 -0.0317467 ]
 [-0.01713343 -0.1330484   0.00426238]
 [-0.00924982  0.0393133   0.00340141]
 [-0.13937329 -0.07896515  0.1223143 ]
 [ 0.20252373  0.11493461  0.1330035 ]
 [-0.10270737 -0.14528474  0.06531235]
 [-0.0045143   0.00425957 -0.11623296]
 [ 0.1996964  -0.00575275  0.06009264]
 [ 0.02558654  0.18013609  0.01821672]
 [-0.05736963  0.08217266  0.05199675]
 [ 0.06332028 -0.03888967  0.01139567]
 [-0.03585111 -0.05301994 -0.01868776]
 [ 0.03614949 -0.0059489  -0.03403357]
 [ 0.06895694 -0.18118275 -0.07975596]
 [-0.02039617 -0.0028409  -0.13670009]
 [-0.1733669  -0.08369999 -0.10551918]
 [-0.00715934  0.08265033  0.03981972]
 [ 0.02268058  0.00804744  0.06603382]
 [ 0.04164357 -0.06604654 -0.01004583]
 [ 0.04702472  0.10251048  0.03000651]
 [-0.06049363 -0.08851413  0.07006089]
 [-0.0464842  -0.12615253  0.08386265]
 [-0.04471621 -0.0605753   0.03616477]
 [-0.0567396   0.04123848  0.00441807]
 [ 0.03573751  0.03358343  0.04329806]
 [-0.02741293 -0.00238602 -0.21201974]
 [ 0.04137417 -0.10870965 -0.04785357]
 [ 0.01153424 -0.03410676 -0.07406767]
 [-0.01379139  0.03623655  0.01595119]
 [-0.04118022  0.07057916 -0.05923101]
 [-0.18754618  0.02879753  0.12460117]
 [ 0.06807989 -0.03721476  0.02859884]
 [-0.22836578  0.0157794  -0.02001626]
 [-0.08744638 -0.05697522  0.0543846 ]
 [ 0.10295055 -0.17644791  0.14807752]
 [-0.04137417  0.10870965  0.04785357]
 [ 0.10371296  0.01720703  0.01753365]
 [-0.07284122  0.14450814 -0.08405568]
 [-0.00747651 -0.07353344  0.15089596]
 [-0.0343815   0.07152613 -0.01366431]
 [-0.07352261  0.14137736 -0.0445318 ]
 [-0.05516555  0.1449462   0.06380476]],

 Films characteristics:
[[ 0.09296504  0.43669991  0.2693743   0.12853428 -0.08870624 -0.83886728]
 [ 0.03512916  0.63524784 -0.76001901  0.04926365 -0.06453129  0.10490964]
 [ 0.45685515 -0.48020396 -0.42360706  0.41995691  0.33327777 -0.30627881]]

 Coef:
[ 9.7660324  10.48691192 13.27822661]
```

### Visualization

Creates two 3D scatter plots:
- **User space**
  <img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/6b178dbb-a2ab-4e9e-bf84-cda0540b3508" />

First 20 users plotted in 3-factor latent space

- **Movie space**
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/d0d33732-10ee-4f7a-9b05-5688399c51f0" />

All movies plotted in 3-factor latent space

---

Task 3: Generating Movie Recommendations
Complete recommendation system that predicts ratings for unseen movies and generates personalized top-10 recommendations.
Algorithm

Load and preprocess ratings data (keep users with 33+ ratings, movies with 44+ ratings)
Fill missing values with 2.5 and demean by user average
Apply SVD with k=3 to decompose the matrix: R = U · Σ · V^T
Reconstruct full rating matrix with predictions
Extract only predicted ratings (originally missing values)
Sort predictions and recommend top-10 highest-rated movies

Code
pythonimport numpy as np
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
Example Output
Top 10 recommendations for user with id 5:

Die Hard: With a Vengeance (1995) — Action|Crime|Thriller
Star Wars: Episode IV - A New Hope (1977) — Action|Adventure|Sci-Fi
Forrest Gump (1994) — Comedy|Drama|Romance|War
Speed (1994) — Action|Romance|Thriller
Jurassic Park (1993) — Action|Adventure|Sci-Fi|Thriller
Silence of the Lambs, The (1991) — Crime|Horror|Thriller
Mission: Impossible (1996) — Action|Adventure|Mystery|Thriller
Independence Day (a.k.a. ID4) (1996) — Action|Adventure|Sci-Fi|Thriller
Star Wars: Episode V - The Empire Strikes Back (1980) — Action|Adventure|Sci-Fi
Star Wars: Episode VI - Return of the Jedi (1983) — Action|Adventure|Sci-Fi



