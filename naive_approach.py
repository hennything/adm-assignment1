import pandas as pd
from sklearn.model_selection import train_test_split

ratings = pd.read_csv('ml-1m/ratings.dat', header=None, sep='::', engine='python',
                      names=["User ID", "Movie ID", "Rating", "Timestamp"])
movies = pd.read_csv('ml-1m/movies.dat', header=None, sep='::', engine='python',
                     names=["Movie ID", "Title", "Genres"])
users = pd.read_csv('ml-1m/users.dat', header=None, sep='::', engine='python',
                    names=["User ID", "Gender", "Age", "Occupation", "Zip-code"])

#print(ratings.head(3))
#print(movies.head(3))
#print(users.head(3))

#train, test = train_test_split(ratings, test_size=0.2)

global_mean = ratings["Rating"].mean()
#print(round(global_mean), 2)

merged_inner_ratings_users = pd.merge(left=ratings, right=users, left_on='User ID', right_on='User ID')

print(merged_inner_ratings_users.head(10))


def get_user_mean(user_id):
    rating_count = 0
    summed_rating = 0

    for index, row in merged_inner_ratings_users.iterrows():
        #print(type(row['User ID']))
        #print(user_id)
        #print(type(user_id))
        if row['User ID'] == user_id:
            rating_count += 1
            summed_rating += row['Rating']

    mean_value = summed_rating / rating_count
    return mean_value

print(get_user_mean(1))

'''
user_average_ratings = dict()

user_rating_dict = (merged_inner_ratings_users.groupby('User ID')
                    .apply(lambda x: list(map(list, zip(get_user_mean(x['User ID'])))))
                    .to_dict())

print(user_rating_dict)


merged_inner_ratings_users_movies = pd.merge(left=merged_inner_ratings_users, right=movies,
                                             left_on='Movie ID', right_on='Movie ID')

'''