
import pandas as pd
import numpy as np 
import os
import json
import scipy.stats as stats
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
sns.set_theme()


#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_sequared_error, r2_score

# Model: Linear Regression
google_data = os.path.join(os.getcwd(), 'Data', 'queryData.csv')
df = pd.read_csv(google_data)

#Explore the data
print(df.head(10))
print(df.shape)

#Clean the data
print("-----Missing data-----")
print(np.sum(df.isnull(), axis = 0))

df_cleaned = df.dropna(subset = ['x__score'])
#Other possibility: use a linear regression to fill in the missing data

print(df_cleaned.head(20))
print(np.sum(df_cleaned.isnull(), axis = 0))



# #Data Visualization
# sns.histplot(data=df_cleaned, x="x__rank")

#Clean the data
print("-----Missing data-----")
print(np.sum(df.isnull(), axis = 0))

df_cleaned = df.dropna(subset = ['x__score'])

# Correlation Values

df1 = df_cleaned.head(5)
correlation1 = df1['x__score'].corr(df['x__rank'])
print(df1)
print(correlation1)

#Other possibility: use a linear regression to fill in the missing data

print(df_cleaned.head(20))
print(np.sum(df_cleaned.isnull(), axis = 0))

#Data Visualization
sns.histplot(data=df_cleaned, x="x__rank")

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