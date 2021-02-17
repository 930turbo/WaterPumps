# import statements
import pandas as pd
import os
import glob
import csv

#--------------------------------#
# GENERATE CSV - UNORGANIZED DATA
#--------------------------------#

# using glob we match the pattern name 'csv' to save the list of file names as the variable 'csvfile'
extension = 'csv'
csvfile = [s for s in glob.glob('*.{}'.format(extension))]

# concatenate all files in given list and export as a single csv file
combined_csv = pd.concat([pd.read_csv(f) for f in csvfile ])
# now export to csv
combined_csv.to_csv( "combineddata.csv", index=False, encoding='utf-8-sig')

#------------------------#
# ORGANIZED TERMINAL DATA
#------------------------#

# generate data frames to view data in terminal
df = pd.read_csv ('testset.csv')
df2 = pd.read_csv ('trainsetlabels.csv')
df3 = pd.read_csv ('trainsetvalues.csv')

# print out data frames in terminal
print(df)
print(df2)
print(df3)

#---------------------------------------#
# PARSE THROUGH NEWLY GENERATED CSV FILE
#---------------------------------------#

# a='non functional'                                      # String we're searching for in the data
# with open("combineddata.csv") as f_obj:
#     reader = csv.reader(f_obj, delimiter=',')
#     for line in reader:                                 # Iterating through rows in our new CSV file
#         print(line)                                     # Line = Rows in CSV file
#         if a in line:                                   # Checks to see if string we are manually searching for is in the row
#             print("String found in row of csv")
