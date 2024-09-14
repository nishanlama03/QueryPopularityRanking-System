import pandas as pd
import numpy as np 
import os
import json
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_sequared_error, r2_score

# Model: Linear Regression
google_data = os.path.join(os.getcwd(), 'Data', 'queryData.csv')
df = pd.read_csv(google_data)
print(df.head(10))


print(df.head(610))

print("hello")

#Clean the data
print("-----Missing data-----")
print(np.sum(df.isnull(), axis = 0))

df_cleaned = df.dropna(subset = ['x__score'])
#Other possibility: use a linear regression to fill in the missing data

print(df_cleaned.head(20))
print(np.sum(df_cleaned.isnull(), axis = 0))


#Detecting outliers

#plot a histogram to see how skewed the data is
plt.figure(figsize=(10, 6))
plt.hist(df['x__score'], bins=30, color='skyblue', alpha=0.7)
plt.title('Histogram of X Score')
plt.xlabel('X Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#Method choosen: winzoriation + z_scores, since we don't want to outliers to affect the dataset
#score: how many times people search

score_999 = np.percentile(df['x__score'], 99.9)

# Winsorize the 'x_score' by capping values at the 99.9th percentile
df['x_score_winsorized'] = np.where(df['x__score'] > score_999, score_999, df['x__score'])

# Calculate Z-scores for the winsorized 'x_score' column
df['zscore_x_score'] = stats.zscore(df['x_score_winsorized'])

df_no_outliers = df[(df['zscore_x_score'] >= -3) & (df['zscore_x_score'] <= 3)]
df_no_outliers.head()

print(max(df['x__rank']))


#score and rank, -.16
#week and rank, -0.6 