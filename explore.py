import pandas as pd

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python')
movies = pd.read_csv('ml-1m/movies.dat', header=None, sep='::', engine='python')
users = pd.read_csv('ml-1m/users.dat', header=None, sep='::', engine='python')
