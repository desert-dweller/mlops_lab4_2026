import pandas as pd
import os

# Fetch Wine Quality dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url, sep=';')

# Create a binary target: 1 if quality >= 7 (Premium), else 0
df['is_premium'] = (df['quality'] >= 7).astype(int)
df = df.drop('quality', axis=1)

os.makedirs('data', exist_ok=True)
df.to_csv('data/wine_data.csv', index=False)
print("Data successfully fetched and preprocessed!")