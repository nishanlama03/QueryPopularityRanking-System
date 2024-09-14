import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt 
import seaborn as sns
#sns.set_theme()

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
#sns.histplot(data=df_cleaned, x="x_rank")


#Data duplicate Deletion
print(df_cleaned[df_cleaned.duplicated()])
# Convert the necessary columns and calculate the 'total' column
df_cleaned['x__rank'] = pd.to_numeric(df_cleaned['x__rank'], errors='coerce')
df_cleaned['x__score'] = pd.to_numeric(df_cleaned['x__score'], errors='coerce')

# Create the 'total' column without warnings
df_cleaned['total'] = (df_cleaned['x__rank'] + df_cleaned['x__score']) / 2

# Convert 'x__week' to datetime
df_cleaned['x__week'] = pd.to_datetime(df_cleaned['x__week'], errors='coerce')

print(df_cleaned[['x__week', 'x__rank', 'x__score', 'total']].head(10))

# Data visualization
plt.figure(figsize=(10,6))
sns.scatterplot(x='x__week', y='x__rank', data=df_cleaned)

# Add labels to each point
for i, row in df_cleaned.iterrows():
    plt.text(row['x__week'], row['x__rank'], str(row['term']), fontsize=9, ha='right')

# Show plot with labels
plt.title('Scatter plot of x__week vs x__rank')
plt.xlabel('x__week')
plt.ylabel('x__rank')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
