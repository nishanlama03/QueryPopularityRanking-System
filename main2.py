import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme()

#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_sequared_error, r2_score

# Model: Linear Regression
google_data = os.path.join(os.getcwd(), 'Data', 'bq-results-20241005-192453-1728156315956.csv')
df = pd.read_csv(google_data)

#Explore the data
print(df.head(10))
print(df.shape)

#Clean the data
print("-----Missing data-----")
print(np.sum(df.isnull(), axis = 0))

# There are very few examples with a value for 'score'
# Instead of dropping examples with missing values, the below line drops the 'score' feature entirely
df = df.drop(columns = ['score'], axis = 1)

# Drop dma_name and dma_id
df = df.drop(columns = ['dma_name', 'dma_id'], axis = 1)

#Other possibility: use a linear regression to fill in the missing data

print(df.head(20))
print(np.sum(df.isnull(), axis = 0))

#Data Visualization
sns.histplot(data=df, x="rank")

#Detecting and Replacing Outliers

# Transform date values to numerical values usable by a model (weeks since earliest data, 2019-08-25)
df['week'] = pd.to_datetime(df['week'])
df['week'] = (df['week'] - pd.Timestamp('2019-07-07')).dt.days
print(df.head(600))

'''
# Plot correlation between score and rank
correlation1 = df['x__score'].corr(df['x__rank'])
print(correlation1)

plt.scatter(df['x__score'], df['x__rank'])
plt.title(f"Correlation Between Score and Rank ({correlation1})")
plt.xlabel('Score')
plt.ylabel('Rank')
plt.show()

# Plot correlation between week and rank
correlation2 = df['x__week'].corr(df['x__rank'])
print(correlation2)

plt.scatter(df['x__week'], df['x__rank'])
plt.title(f"Correlation Between Week and Rank ({correlation2})")
plt.xlabel('Week')
plt.ylabel('Rank')
plt.show()
'''

# Find the number of unique values in 'term' column
unique_vals = df['term'].nunique()
print(f"Unique values in 'term' column: {unique_vals}")

# Perform one-hot encoding
df = pd.get_dummies(df, columns = ['term'], drop_first = True)
print(df.head())
print(df.shape)
