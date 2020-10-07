import pandas as pd 
import numpy as np
from numpy.linalg import matrix_rank

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python',
                        names=["User ID", "Movie ID", "Rating", "Timestamp"])

utility_matrix = ratings.pivot(index='User ID', columns='Movie ID', values='Rating')

# print(matrix_rank(utility_matrix))

print(len(ratings)/(utility_matrix.shape[0]*utility_matrix.shape[1]))

print(utility_matrix.isnull().sum().sum()/(utility_matrix.shape[0]*utility_matrix.shape[1]))