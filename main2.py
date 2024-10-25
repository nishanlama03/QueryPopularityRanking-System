import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt 
import seaborn as sns
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_sequared_error, r2_score

# Model: Linear Regression
google_data = os.path.join(os.getcwd(), 'Data', 'bq-results-20241005-192453-1728156315956.csv')
df = pd.read_csv(google_data)

#Explore the data
print(df.head(10))
print(df.shape)

# Delete first 5,000,000 rows of the dataframe
print("Deleting rows...");
df = df.iloc[8000000:]
print(df.shape)

#Clean the data
print("-----Missing data-----")
nan_count = np.sum(df.isnull(), axis = 0)

# There are very few examples with a value for 'score'
# Instead of dropping examples with missing values, the below line drops the 'score' feature entirely
# df = df.drop(columns = ['score'], axis = 1)

# Drop refresh_date column (all values are equal)
df = df.drop(columns = ['refresh_date'], axis = 1)

# Drop dma_name and dma_id
df = df.drop(columns = ['dma_name', 'dma_id'], axis = 1)

#Other possibility: use a linear regression to fill in the missing data

condition = nan_count != 0
col_names = nan_count[condition].index
col_names = list(col_names)
print(df['score'])
for i in col_names:
    df[i].fillna(value = -1, inplace = True)

print(df.head(20))
print(np.sum(df.isnull(), axis = 0))

#Data Visualization
sns.histplot(data=df, x="rank")

#Detecting and Replacing Outliers

# Transform date values to numerical values usable by a model (weeks since earliest data, 2019-08-25)
df['week'] = pd.to_datetime(df['week'])
df['week'] = (df['week'] - pd.Timestamp('2019-07-07')).dt.days
print("df with transformed dates...")
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

# Convert one-hot encoded columns to numerical values
bool_columns = df.select_dtypes(include=['bool']).columns.tolist()

# Convert boolean columns to integers one at a time (to prevent a memory spike)
for col in bool_columns:
    df[col] = df[col].astype(int)   

print(df.head())

# Standardize feature columns
label = df['rank']
scaler = StandardScaler()
standardized = scaler.fit_transform(df.drop(columns = ['rank'], axis = 1))
standardized_features = pd.DataFrame(standardized, columns = df.drop(columns = ['rank'], axis = 1).columns)
standardized_df = pd.concat([standardized_features, label.reset_index(drop = True)], axis = 1)
df = standardized_df
print(df.head())

# Assign features and label variables
y = df['rank']
X = df.drop(columns = ['rank'], axis = 1)

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# # Train a simple linear model
# modelLR = LinearRegression()
# modelLR.fit(X_train, y_train)
# # Evaluate on the test set
# test_score = modelLR.score(X_test, y_test)
# print("Test R-squared score:", test_score)

modelDT = DecisionTreeRegressor(random_state = 1234)

param_grid = {
    'criterion': ['friedman_mse'], 
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_leaf': [None, 1, 2, 5, 10]
}

grid_search = GridSearchCV(estimator = modelDT, param_grid = param_grid, cv = 5, scoring = 'neg_mean_squared_error', verbose = 1, n_jobs = -1)
print("1")
grid_search.fit(X_train, y_train)
print("2")
best_model = grid_search.best_estimator_
print("3")
test_score = modelDT.score(X_test, y_test)
print("4")
print("Test R-squared score:", test_score)