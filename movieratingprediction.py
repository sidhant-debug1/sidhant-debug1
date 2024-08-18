import numpy as np 
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Update this path to where your CSV file is located
file_path = "IMDb Movies India.csv"

# Try reading the CSV file with a specified encoding
try:
    data = pd.read_csv(file_path, encoding='ISO-8859-1')  # You can try other encodings if this doesn't work
except Exception as e:
    print(f"An error occurred: {e}")
    exit()  # Exit if the file is not found or cannot be read

# Initial data exploration
print('Number of Rows:', data.shape[0])
print('Number of Columns:', data.shape[1])
data.info()

# Check for null values
print(data.isnull().sum())

# Drop rows with missing values in essential columns
data = data.dropna(subset=['Duration', 'Votes', 'Rating', 'Year', 'Genre'])

# Convert 'Duration' to numeric (remove non-numeric characters and convert to int)
data['Duration'] = data['Duration'].str.extract('(\d+)').astype(float)

# Create directory for saving plots
plot_dir = "plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Plot Votes by Year
plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Votes', data=data)
plt.title("Votes By Year")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "votes_by_year.png"))
plt.show()

# Plot Top 10 Longest Movies
top_10_longest = data.nlargest(10, 'Duration')[['Name', 'Duration']].set_index('Name')
plt.figure(figsize=(10, 6))
sns.barplot(x='Duration', y=top_10_longest.index, data=top_10_longest)
plt.title('Top 10 Longest Movies')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "top_10_longest_movies.png"))
plt.show()

# Plot Number of Movies Per Year
plt.figure(figsize=(10, 6))
sns.countplot(x='Year', data=data)
plt.title("Number of Movies Per Year")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "number_of_movies_per_year.png"))
plt.show()

# Plot Top 10 Highest Rated Movies
top_10_rated = data.nlargest(10, 'Rating')[['Name', 'Rating', 'Director']].set_index('Name')
plt.figure(figsize=(10, 6))
sns.barplot(x='Rating', y=top_10_rated.index, data=top_10_rated)
plt.title("Top 10 Highest Rated Movies")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "top_10_highest_rated_movies.png"))
plt.show()

# Plot Average Rating by Year
data1 = data.groupby('Year')['Rating'].mean().sort_values(ascending=False).reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Rating', data=data1)
plt.title("Average Rating by Year")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "average_rating_by_year.png"))
plt.show()

# Define rating categories
def rating_category(rating):
    if rating >= 7.0:
        return 'Excellent'
    elif rating >= 6.0:
        return 'Good'
    else:
        return 'Average'

data['rating_cat'] = data['Rating'].apply(rating_category)
print(data.head(1))

# Genre count
data['Genre'] = data['Genre'].str.split(',')
genre_counts = pd.Series([genre for sublist in data['Genre'] for genre in sublist]).value_counts()

# Plot Genre counts
plt.figure(figsize=(12, 8))
sns.barplot(x=genre_counts.values, y=genre_counts.index)
plt.title('Genre Counts')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "genre_counts.png"))
plt.show()
