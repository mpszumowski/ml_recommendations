import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances


header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
# print(df)
print(n_users)
print(n_items)
# The dataframe is a matrix of 100000 rows and 4 columns: pk, user_id,
# item_id and rating. There are 943 unique users and 1682 unique movies
# of which there are 1000000 ratings.
# The next step is to split the dataset into train and test sets. I will omit
# the k-fold cross-validation. Then I will create two matrices for each set
# with m-rows of unique users and n-columns of unique movies displaying
# their ratings given by each user. The matrices will be sparse

# Divide the dataset into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.25)
print('train_data length: {}'.format(len(train_data)))
print('test_data length: {}'.format(len(test_data)))


#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
# unpack the Pandas object
for line in train_data.itertuples():
    # adjust to count rows and cols from 0 and fill in the matrix
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

# print(train_data_matrix.shape)
# Calculate the cosine similarity and return it as matrices user x user
# and item x item.
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
# print(user_similarity)


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')


