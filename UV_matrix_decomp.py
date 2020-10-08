import pandas as pd
import numpy as np

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python',
                        names=["User ID", "Movie ID", "Rating", "Timestamp"])

print(len(ratings))

user_IDs = ratings['User ID'].unique()
movie_IDs = ratings['Movie ID'].unique()

print('There are in total ' + str(len(movie_IDs)) + ' movies in the database')
print('There are in total ' + str(len(user_IDs)) + ' users in the database')

M = pd.DataFrame(index=user_IDs, columns=movie_IDs)

print(M.isnull().sum().sum())

# print(len(movie_IDs)*len(user_IDs))

for index, row in ratings.iterrows():
    current_row = M.iloc[row['User ID'] - 1]
    current_row[row['Movie ID']] = row['Rating']

U = pd.DataFrame(np.random.randint(1, 1, size=(len(user_IDs), 2)))
V = pd.DataFrame(np.random.randint(1, 1, size=(2, len(movie_IDs))))

#create matric same shape as M, to hold the predicted values, so it's just UV
UV = pd.DataFrame(np.random.randint(1, 1, size=(len(user_IDs), len(movie_IDs))))

#walk through all elements of UV to predict their value
for i in range(0, len(user_IDs)):
    for j in range(0, len(movie_IDs)):
        UV_current = UV[i][j]

        summed_value = 0

        #walk through elements of U for same row as UV_current
        #and V for same column as UV_current
        for n in range(0, len(movie_IDs)):
            U_current = U[i][n]
            V_current = V[n][j]

            summed_value += U_current * V_current

        UV_current[i][j] = summed_value

#UV is now entirely filled, and its value can be compared to M


#find out how close to M the product of U and V is
def rmse(UV, U, V, M):
    element_RMSE = 0

    #sum over non-blank entries of M
    #take the mean of each element by dividing by the number of non-blank elements summed
    #take the square of the mean

    return element_RMSE







