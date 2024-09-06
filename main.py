import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_sequared_error, r2_score

file_path = 'queryData.json'
df = pd.read_json(file_path, lines=True)

print("hello")