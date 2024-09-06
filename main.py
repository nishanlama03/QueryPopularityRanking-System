import pandas as pd
import numpy as np
import os
import json

#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_sequared_error, r2_score

google_data = os.path.join(os.getcwd(), 'Data', 'queryData.csv')
df = pd.read_csv(google_data)
print(df.head(10))

print("hello")