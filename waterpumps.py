# import statements
import pandas as pd

# generate data frames to view data
df = pd.read_csv ('testset.csv')
df2 = pd.read_csv ('trainsetlabels.csv')
df3 = pd.read_csv ('trainsetvalues.csv')

# print out data frames
print(df)
print(df2)
print(df3)
