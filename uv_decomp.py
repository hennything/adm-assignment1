import pandas as pd 
import numpy as np
from math import sqrt

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python',
                        names=["User ID", "Movie ID", "Rating", "Timestamp"])

utility_matrix = ratings.pivot(index='User ID', columns='Movie ID', values='Rating')

m, n = utility_matrix.shape

# keeping the number of non-blank entries as a variable 
nr_non_blanks = utility_matrix.notna().sum().sum()
print(nr_non_blanks)

# Global Average rating for the existing ratings
# not very pretty and too many chained functions.
glb_avg = utility_matrix.sum().sum() / utility_matrix.notna().sum().sum()

# U and V should not be filled with the gloabl average but rather the values which when multiplied equal the global average (non-blank elements). 
U = np.full((m, 2), sqrt(glb_avg))
V = np.full((2, n), sqrt(glb_avg))

# U, V = np.linalg.qr(UV, mode='reduced')

# print(U.shape, V.shape, UV.shape)

# print(U*V.T == UV)

UV = U.dot(V)

print(UV)

# print(sqrt((utility_matrix-UV).sum().sum()/nr_non_blanks))

# print(utility_matrix)

# print(UV)




# print(len(ratings)/(utility_matrix.shape[0]*utility_matrix.shape[1]))
# print(utility_matrix.isnull().sum().sum()/(utility_matrix.shape[0]*utility_matrix.shape[1]))