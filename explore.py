import pandas as pd
import matplotlib.pyplot as plt

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python',
                      names=["User ID", "Movie ID", "Rating", "Timestamp"])
movies = pd.read_csv('ml-1m/movies.dat', header=None, sep='::', engine='python',
                     names=["Movie ID", "Title", "Genres"])
users = pd.read_csv('ml-1m/users.dat', header=None, sep='::', engine='python',
                    names=["User ID", "Gender", "Age", "Occupation", "Zip-code"])



# plt.hist(ratings.Rating, bins=10)
# plt.show()

print(ratings.head(3))
print(movies.head(3))
print(users.head(3))

