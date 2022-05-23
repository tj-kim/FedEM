import pandas as pd
import numpy as np
import sys
import os

np.random.seed(123)

args = sys.argv

# args[0] -> whole dataset file path
print(f"File path {args[1]}")
print(args)
df = pd.read_csv(args[1])

# args[1] -> number of clients
cnt = len(df.userId.unique())
clients = int(args[2])

idx = np.random.permutation(cnt)
df_list = []
num_per_cl = cnt//clients
path = "all_data/train/task_{client}"
ratio = 0.8

unique_users = df.userId.unique()
user_to_index = {old: new for new, old in enumerate(unique_users)}
new_users = df.userId.map(user_to_index)
    
unique_movies = df.movieId.unique()
movie_to_index = {old: new for new, old in enumerate(unique_movies)}
new_movies = df.movieId.map(movie_to_index)

unique_ratings = df.rating.unique()
rating_to_index = {old: new for new, old in enumerate(unique_ratings)}
new_ratings = df.rating.map(rating_to_index)
print(unique_ratings)

n_users = unique_users.shape[0]
n_movies = unique_movies.shape[0]

print(n_users)
print(n_movies)

X = pd.DataFrame({'userId': new_users, 'movieId': new_movies, 'rating': new_ratings})
#y = ratings['rating'].astype(np.float32)

for cl in range(clients):
    mn = cl*num_per_cl
    mx = min((cl+1)*num_per_cl, cnt)
    indices = idx[mn:mx]
    num_data_points = len(indices)
    train_indices = indices[0:int(ratio*num_data_points)]
    test_indices = indices[int(ratio*num_data_points):]
    directory = path.format(client=cl) 
    if not os.path.exists(directory):
        os.makedirs(directory)
    



    X.loc[X['userId'].isin(train_indices)].to_csv(os.path.join(directory,"train.csv"))
    X.loc[X['userId'].isin(test_indices)].to_csv(os.path.join(directory,"test.csv"))
