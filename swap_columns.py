import pandas as pd
import os

train_path = r'c:\Users\meagie\Desktop\chronically-online-translator\data\processed\train.csv'
test_path = r'c:\Users\meagie\Desktop\chronically-online-translator\data\processed\test.csv'

# Read train.csv
train_df = pd.read_csv(train_path)
print("Train.csv BEFORE swap:")
print(train_df.head(2))
print()

# Swap columns
train_df = train_df[['target_text', 'source_text']]
train_df.columns = ['source_text', 'target_text']

# Save
train_df.to_csv(train_path, index=False)
print("Train.csv AFTER swap - saved successfully")
print()

# Read test.csv
test_df = pd.read_csv(test_path)
print("Test.csv BEFORE swap:")
print(test_df.head(2))
print()

# Swap columns
test_df = test_df[['target_text', 'source_text']]
test_df.columns = ['source_text', 'target_text']

# Save
test_df.to_csv(test_path, index=False)
print("Test.csv AFTER swap - saved successfully")
