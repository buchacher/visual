import pandas as pd
from matplotlib import pyplot as plt

# Load data using pandas
df = pd.read_csv('data/data_train.csv')

'''Uncomment below to plot color class distribution and output value counts'''
# plt.hist(df['color'])
# plt.title("Color Class Distribution")
# plt.show()
# print(df['color'].value_counts())

'''Uncomment below to plot texture class distribution and output value counts'''
# plt.hist(df['texture'])
# plt.title("Texture Class Distribution")
# plt.show()
# print(df['texture'].value_counts())

'''Uncomment below to print all column names'''
# for col in df.columns:
#     print(col)
