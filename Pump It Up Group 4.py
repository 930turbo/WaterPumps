#__________ ____ ___  _____ __________  .______________  ____ _____________ 
#\______   \    |   \/     \\______   \ |   \__    ___/ |    |   \______   \
# |     ___/    |   /  \ /  \|     ___/ |   | |    |    |    |   /|     ___/
# |    |   |    |  /    Y    \    |     |   | |    |    |    |  / |    |    
# |____|   |______/\____|__  /____|     |___| |____|    |______/  |____|    
#                          \/                                               
#  ________                            ___________                  
# /  _____/______  ____  __ ________   \_   _____/___  __ _________ 
#/   \  __\_  __ \/  _ \|  |  \____ \   |    __)/  _ \|  |  \_  __ \
#\    \_\  \  | \(  <_> )  |  /  |_> >  |     \(  <_> )  |  /|  | \/
# \______  /__|   \____/|____/|   __/   \___  / \____/|____/ |__|   
#        \/                   |__|          \/                     

# Giancarlo Coelho - Derek Smith - Brendan Mandile

# Whats really special about this project is that the data in theory could be used to actually HELP people instead of most generic programming questions for competitions.
# When we first approched this project we wanted to draw out a series of steps that would allow us to consicely organize and sort the data allowing for the creation of different visuals. 
# The first step was, naturally, to create out import statements and link the files given to us so we can call on them in out next steps. Originally we were going to use ggplot, but after some consideration we decided to use the seaborn library due to its multitude of graphics options for data visualization. 

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mlt
import matplotlib.image as mpimg

# Using Jupyter we are able to live view contents of files which is great--here we imported all of the files so we can start the project. 
trainvalues_df = pd.read_csv ('C:/Users/student/Documents/Data/TrainingSetValues.csv')
trainlabels_df = pd.read_csv ('C:/Users/student/Documents/Data/TrainingSetLabels.csv')
testvalues_df = pd.read_csv ('C:/Users/student/Documents/Data/TestSetValues.csv')
subform_df = pd.read_csv ('C:/Users/student/Documents/Data/SubmissionFormat.csv')

trainvalues_df.head()
trainlabels_df.head()
testvalues_df.head()
#subform_df.head()


# %%
# Seeing how much data we're working with we don't want to choke the system. There are thousands of values and some may become modified so we'll set the columns to an unlimited amount in pandas.

pd.set_option('max_columns', None)


# %%
# By using pandas we can use df.describe to generate statistics for our files. Pandas is great in the sense that it will analyze mixed data types, numbers, and objects in series and will summarize different aspects such as tendencies, dispertions, and distributions...all while excluding null values. Plus one for pandas. 

train_df.describe()


# %%
# As we were arrived at this step it was obvious that the data(s) were still not interacting with each other. To fix this we took the two most similar values that spoke hand to hand with each other--train values and train lables

# For shorthand sake we'll be using trn instead of train when working with our dataframe. 

trn_df = trainvalues_df.merge(trainlabels_df, on='id')

trn_df


# %%
# So to quickly explain whats going on here we want to eliminate as many null values from our data as we can. As you'll see later on in this project some of the water pumps are found in the middle of the ocean, we're trying to avoid as much of that type of data as possible. There are 40 different ID names we're considering so we want to make sure we're including all 40. By using df.info we're printing a summary of what the combined data frame contains which will allow us to see our ID names on an organized list. 

trn_df.info()
print('_'*40)

# %% [markdown]
# And here I will look at categorical data and see if it makes sense to keep it all. I want to look at low frequency occurrences because they often highlight bad data.

# %%
# In our next step here we want to go through each ID from the training file and make sure that what we'll proceed with is exactly what we want. The thought being that data that has low occurances with itself or that cannot provide enough usable data will lead to that data being 'bad' and in turn will take up more memory for no reason. By cutting down on what data we'll be using we can have an easier time down the road. 
print(trainvalues_df.columns.values('funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'region', 'lga', 'ward', 'public_meeting', 'recorded_by', 'scheme_management', 'scheme_name', 'permit', 'construction_year', 'extraction_type', 'extraction_type_group', 'extraction_type_class', 'management', 'management_group', 'payment', 'payment_type', 'water_quality', 'quality_group', 'quantity','quantity_group', 'source', 'source_type','source_class','waterpoint_type','waterpoint_type_group')

trn_df.describe(include=['O'])

# %% [markdown]
# Maybe I'll run a correlation matrix so I can find a) potential good predictors and b) highly similar columns that means I might want to use just one of the columns.

# %%
# The next logical step in our program would be to run a correlation matrix so we can attempt to find good predictors in the data and also decide on the number of columns we'll be using. 

trn_df.lga.value_counts()


# %%
# Using a little help from the pandas docs we were then able to figure out the best way of writing the next section. By running this snippet we're returning a subset of the data. An example of this snippet would be like assigning numbers to values, this will organize and display the data in a readable fashion making life much easier.

obj_df = trn_df.select_dtypes(include=['object']).copy()

obj_df.head()


# %%
status_organize = {"status_group":     {"non functional": 0, "functional needs repair": 1, "functional": 2}}
 
trn_df = trn_df.replace(status_organize)
trn_df.head()
    


# %%
# A consideration to have here would be to create another grouping of data where we examine construction date of each pump and then visualize it to see if we can pick up on patterns such as life of a pump per defined X and Z distance. 

# -----------
#g = sns.FacetGrid(tips, col="Age",  row="Operation Ability")
#g.map(sns.scatterplot, "construction_year", "age")
# -----------

# Figsize values are floats, we're organizing gps data now so we need to be able to numerical represent lat and long data to a decimal. 

corMat = trn_df.corr()
print(corMat)

# By making the size of the figure 10 X 10 we're creating a heatmap that will be 1040 x 1040 pixels. More than enough size to visualize this data with good detail. I also wanted to make the size 1040 because to this day I'm still upset that Nvidia never came out with that card. Would have been an amazing bridge between the 1030 and 1050. 1040 is also half of 2080 which was one of the graphics cards used in my favorite computer I ever built for a client. Had to have something be an identifier of my work :) - Gianni

# The plan here is to annotate our data with numeric values shown on a heatmap. 

fig, axs = plt.subplots(figsize=(13,13))
sns.heatmap(corMat, annot=True, fmt="d")

# Now for a future push the consideration would be to add in a legend, to do this we would just do something like this:

#plt.figure(figsize=(13,13))
#sns.scatterplot(x="x_value_about_pumps", 
#                y="y_value_about_pumps", 
#                hue="z",                      ----- This would be based on whatever we would define up above based on which information we'd be feeding it.
#                data=penguins_df)
#plt.xlabel("x_value_about_pumps")
#plt.ylabel("y_value_about_pumps")

# I'll be completely honest it would take me reading up on seaborn a little to move the legend outside of the graph itself but I'm assuming it would be something along the lines of defining it as an image with a dpi rating and then anchoring it somewhere outside the graphs border. 

# %% [markdown]
# So our next thought related directly to the environment and this is why we wanted to include the GPS data from the beginning. The thought occured to us that at higher altitudes there is less pressure thus causing for premature pump failure by overworking the machines. 

# %%
# Height directly correlated to altitude. Interchangable words. 

trn_df[['gps_height', 'status_group']].groupby(['status_group'], as_index=False).mean().sort_values(by='gps_height', ascending=False)

# %% [markdown]
# By organizing each region by a code we can easier sort each section of the map. As we hypothesized the higher locations on the map were prone to failture. There was a very strong difference in different functioning levels of each pump. From this state we just have to organize latitudinal and longitudinal data to create the true map.

# %%
# Sorting by area codes starting with the larger one which is the entire region. 

trn_df[['region_code', 'status_group']].groupby(['region_code'], as_index=False).mean().sort_values(by='status_group', ascending=False)
trn_df[['district_code', 'status_group']].groupby(['district_code'], as_index=False).mean().sort_values(by='status_group', ascending=False)


# %%
trn_df.groupby(['quantity_group','quantity']).count()


# %%
# This code from up above we bring down and modify so we can create multiple instances of the plots.

plt.figure(figsize=(13,13))
axs = sns.countplot(x='quantity',hue='status_group', data=trn_df) # here we use that hue definition mentioned up above

# %% [markdown]
# Now our personal favorite part begins, creating the map of pump placement. 

# %%
# For the install one of us used conda and two of us used pip

#conda install geopandas
#--------
#git clone https://github.com/geopandas/geopandas.git
#cd geopandas
#pip install .
#--------

#import geopandas
#import geoplot

coords = trn_df[['latitude','longitude']]
coords.head()


# %%
# To remove *most* outliers we need to create a plot area. 

pumps = trn_df[(trn_df['longitude'] >= 25)]

pumps.plot(kind="scatter", x="longitude", y="latitude", alpha=1.0)


# %%
# Using image of Tanzania as the basis for our plotting.

tanzania_img=mpimg.imread('C:/Users/student/Documents/Data/tanzania.jpg')

map = pumps.plot(kind="scatter", x="longitude", y="latitude",
    label="pump",c="status_group", cmap=plt.get_cmap("jet"),
    colorbar=True, alpha=1.0, figsize=(13,13),
)
plt.ylabel("latitude", fontsize=13)
plt.xlabel("longitude", fontsize=13)

plt.legend()
plt.show()


# %%
maplabels_df = pd.read_csv ('C:/Users/student/Documents/Data/TrainingSetLabels.csv',nrows = 500)
map_data_df = pd.read_csv ('C:/Users/student/Documents/Data/TrainingSetValues.csv',nrows = 500)
maptrn = map_data_df.merge(maplabels_df, on='id')
maptrn
status_organize = {"status_group":     {"non functional": 0, "functional needs repair": 1, "functional": 2}}
 
map_df = maptrn.replace(status_organize)
map_df.head()


# %%
# The fun part now, folium maps!

import folium

wm = folium.Map(location=[30, -5], zoom_start=3)

tooltip = "Click Here"

# We got a little stuck on this section so thanks to stack overflow we're up and running. 

for index, pump in map_df.iterrows():
    location = [pump['latitude'], pump['longitude']]
    folium.Marker(location, popup = f'ID:{pump["id"]}\n Status:{pump["status_group"]}', tooltip=tooltip).add_to(wm)

wm


# %%
This next section covers decision trees:


# %%
from sklearn.tree import DecisionTreeClassifier # Need to run the classifier to create a tree
from sklearn import tree # Need to run to be able to visualize and plot the tree

dectree_df = trn_df[['status_group', 'gps_height', 'payment_type', 'longitude', 'latitude', 'quantity', 'quality_group',
                    'scheme_management', 'extraction_type','source','waterpoint_type','region','amount_tsh','population']]
                    # This is all of our IDs to compare
dectree_df.head() # This is organizing and displaying everything


# %%
# Due to the fact there are no numerical values assigned we must create a spread of values that will be placeholders. We're using get_dummies for this because it converts numerical data to dummy values. Using same IDs.
dectree_df = pd.get_dummies(dectree_df, columns=["payment_type", 'quantity', 'quality_group',
                                                'extraction_type', 'scheme_management','source','waterpoint_type','region'])


# %%
dectree_df.head()


# %%
X_train = dectree_df.drop("status_group", axis=1)
Y_train = dectree_df.status_group

feature_names = X_train.columns
labels = ['Non Functional', 'Functional Needs Repair', 'Functional']


# %%
clf = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 50)

# Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)

clf.score(X_train, Y_train)


# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(50,30)) # Generate figure size to a readable scale 
tree.plot_tree(clf, fontsize=13, feature_names = feature_names, class_names = labels);
plt.show()


# %%