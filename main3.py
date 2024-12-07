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
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_sequared_error, r2_score


# Model: Linear Regression
google_data = os.path.join(os.getcwd(), 'Data', 'bq-results-20241005-192453-1728156315956.csv')
df = pd.read_csv(google_data)


#Explore the data
print(df.head(10))
print(df.shape)


# Delete first 9,500,000 rows of the dataframe
print("Deleting rows...");
df = df.iloc[9500000:]
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


# Sort data by 'week' in ascending order to maintain chronological order
df = df.sort_values(by='week')


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


# # Assign features and label variables
# y = df['rank']
# X = df.drop(columns = ['rank'], axis = 1)


# # Split training and testing data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# # Train a simple linear model
# model = LinearRegression()
# model.fit(X_train, y_train)


# # Evaluate on the test set
# test_score = model.score(X_test, y_test)
# print("Test R-squared score:", test_score)


# # data visualization


# plt.figure(figsize=(10, 6))
# plt.scatter(df['score'], df['rank'], alpha=0.5)
# plt.title("Correlation Between Score and Rank")
# plt.xlabel("Score")
# plt.ylabel("Rank")
# plt.show()


# #distribution of rank


# plt.figure(figsize=(10, 6))
# sns.histplot(df['rank'], bins=30, kde=True, color='skyblue')
# plt.title("Distribution of Rank")
# plt.xlabel("Rank")
# plt.ylabel("Frequency")


# Determine the index for the split, e.g., 70% train and 30% test
split_index = int(len(df) * 0.7)


# Split data into training and testing sets based on the sorted order
X_train, X_test = df.drop(columns=['rank']).iloc[:split_index], df.drop(columns=['rank']).iloc[split_index:]
y_train, y_test = df['rank'].iloc[:split_index], df['rank'].iloc[split_index:]


# Train and evaluate simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Simple Regression R² Score (Test):", test_score)


# # Define the parameter grid for alpha
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}


# Ridge Regression
print("Running Grid Search for Ridge Regression...")
ridge_model = Ridge()
ridge_grid = GridSearchCV(
   estimator = ridge_model,
   param_grid = param_grid,
   scoring = make_scorer(r2_score),
   cv=5,  # 5-fold cross-validation
   n_jobs = -1
)
ridge_grid.fit(X_train, y_train)


print("Best Ridge Model:", ridge_grid.best_estimator_)
print("Best Ridge R² Score (Train):", ridge_grid.best_score_)
ridge_test_r2 = ridge_grid.best_estimator_.score(X_test, y_test)
print("Ridge R² Score (Test):", ridge_test_r2)


# Lasso Regression
print("\nRunning Grid Search for Lasso Regression...")
lasso_model = Lasso()
lasso_grid = GridSearchCV(
   estimator = lasso_model,
   param_grid = param_grid,
   scoring = make_scorer(r2_score),
   cv = 5,  # 5-fold cross-validation
   n_jobs = -1
)
lasso_grid.fit(X_train, y_train)


print("Best Lasso Model:", lasso_grid.best_estimator_)
print("Best Lasso R² Score (Train):", lasso_grid.best_score_)
lasso_test_r2 = lasso_grid.best_estimator_.score(X_test, y_test)
print("Lasso R² Score (Test):", lasso_test_r2)


# # ElasticNet Regression
print("\nRunning Grid Search for ElasticNet Regression...")
elastic_net_model = ElasticNet()
elastic_net_grid = GridSearchCV(
   estimator = elastic_net_model,
   param_grid = param_grid,
   scoring = make_scorer(r2_score),
   cv=5,  # 5-fold cross-validation
   n_jobs = -1
)
elastic_net_grid.fit(X_train, y_train)


print("Best ElasticNet Model:", elastic_net_grid.best_estimator_)
print("Best ElasticNet R² Score (Train):", elastic_net_grid.best_score_)
elastic_net_test_r2 = elastic_net_grid.best_estimator_.score(X_test, y_test)
print("ElasticNet R² Score (Test):", elastic_net_test_r2)


# # DecisionTree Regression
modelDT = DecisionTreeRegressor(random_state=1234)


param_grid = {
   'criterion': ['friedman_mse'],
   'max_depth': [5, 10, 15, 20],
   'min_samples_leaf': [1, 2, 5, 10]
}


grid_search = GridSearchCV(
   estimator=modelDT,
   param_grid=param_grid,
   cv=5,
   scoring = make_scorer(r2_score),
   verbose=1,
   n_jobs=-1
)
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_
dt_test_score = best_model.score(X_test, y_test)
print("Best DecisionTree Model Details:")
print(f"Criterion: {best_model.criterion}")
print(f"Max Depth: {best_model.max_depth}")
print(f"Min Samples Leaf: {best_model.min_samples_leaf}")
print("Test R-squared score:", dt_test_score)

# knn_param_grid = dict(n_neighbors = [int(x) for x in np.linspace(2, np.sqrt(X_train.shape[0]), num = 10)])
# model_knn = KNeighborsRegressor()
# grid_knn = GridSearchCV(model_knn, knn_param_grid, cv = 5)
# grid_knn_search = grid_knn.fit(X_train, y_train)
# best_k = grid_search.best_params_['n_neighbors']
# best_model = KNeighborsRegressor(best_k)
# best_model.fit(X_train, y_train)
# predictions_best_model = best_model.predict(X_test)
# print(r2_score(y_test, predictions_best_model))

model_knn = KNeighborsClassifier(n_neighbors = 3)
model_knn.fit(X_train, y_train)
class_label_predictions = model_knn.predict(X_test)
print(accuracy_score(y_test, class_label_predictions))

model_rf = RandomForestRegressor(max_depth = 3, n_estimators = 10)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared = False)
print(r2_score(y_test, y_pred_rf))