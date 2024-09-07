import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme()

#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_sequared_error, r2_score

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

#Data Visualization
sns.histplot(data=df_cleaned, x="x_rank")


#Detecting and Replacing Outliers