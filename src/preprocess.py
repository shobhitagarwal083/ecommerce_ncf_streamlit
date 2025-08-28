import pandas as pd

# Load ratings data
ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
movies  = pd.read_csv("data/ml-latest-small/movies.csv")


print("✅ Dataset Loaded")
print(ratings.head())

# Basic preprocessing: Keep only userId, movieId, rating
ratings = ratings[['userId', 'movieId', 'rating']]

# Save processed version
ratings.to_csv("data/processed_ratings.csv", index=False)
print("✅ Processed ratings saved to ../data/processed_ratings.csv")
print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)
