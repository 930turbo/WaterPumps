# import statements
import pandas as pd
import os
import glob

# using glob we match the pattern name 'csv' to save the list of file names as the variable 'csvfile'
extension = 'csv'
csvfile = [i for i in glob.glob('*.{}'.format(extension))]

# concatenate all files in given list and export as a single csv file
combined_csv = pd.concat([pd.read_csv(f) for f in csvfile ])
# now export to csv
combined_csv.to_csv( "combineddata.csv", index=False, encoding='utf-8-sig')

# generate data frames to view data in terminal
df = pd.read_csv ('testset.csv')
df2 = pd.read_csv ('trainsetlabels.csv')
df3 = pd.read_csv ('trainsetvalues.csv')

# print out data frames in terminal
print(df)
print(df2)
print(df3)
