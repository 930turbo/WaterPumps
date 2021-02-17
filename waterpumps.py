import pandas as pd

df = pd.read_csv ('testset.csv')
df2 = pd.read_csv ('trainsetlabels.csv')
df3 = pd.read_csv ('trainsetvalues.csv')

print(df)
print(df2)
print(df3)