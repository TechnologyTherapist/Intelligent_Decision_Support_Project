import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # for visualization

# %matplotlib inline
import seaborn as sns  # for visualization
import warnings

# Data Import
df = pd.read_csv("/Volumes/DATA/Data Analyst Projects/Intelligent_Decision_Support_Project/data/AB_NYC_2019.csv")

df.head()

# Data Exploration and Data Cleaning
df.info()
df.head().T

# checking what are the variables here:
df.columns

# so now first rename few columns for better understanding of variables.
rename_col = {
    "id": "listing_id",
    "name": "listing_name",
    "number_of_reviews": "total_reviews",
    "calculated_host_listings_count": "host_listings_count",
}

# use a pandas function to rename the current function
df = df.rename(columns=rename_col)
df.head(2)

# checking shape of Airbnb dataset
df.shape

# So, host_name, neighbourhood_group, neighbourhood and room_type fall into categorical variable category.

# While host_id, latitude, longitude, price, minimum_nights, number_of_reviews, last_review, reviews_per_month, host_listings_count, availability_365 are numerical variables

# check duplicate rows in dataset
df = df.drop_duplicates()
df.count()

# checking null values of each columns
df.isnull().sum()
df.head()

df["listing_name"].fillna("unknown", inplace=True)
df["host_name"].fillna("no_name", inplace=True)

# so the null values are removed
df[["host_name", "listing_name"]].isnull().sum()

df = df.drop(
    ["last_review"], axis=1
)  # removing last_review column beacause of not that much important

df.info()  # the last_review column is deleted

# The reviews_per_month column also containing null values and we can simple put 0 reviews by replacing NAN's


df["reviews_per_month"] = (
    df["reviews_per_month"].replace(to_replace=np.nan, value=0).astype("int64")
)

# the null values are replaced by 0 value
df["reviews_per_month"].isnull().sum()

# so now check Dataset columns changed and null values, last_review column removed.
df.sample(5)

# Check Unique Value for variables and doing some experiments

# check unique values for listing/property Ids
# all the listing ids are different and each listings are different here.
df["listing_id"].nunique()
# so there are 221 unique neighborhood in Dataset
df["neighbourhood"].nunique()
# and total 5 unique neighborhood_group in Dataset
df["neighbourhood_group"].nunique()
# so total 11453 different hosts in Airbnb-NYC
df["host_name"].nunique()
# most of the listing/property are different in Dataset
df["listing_name"].nunique()

# Note: - so i think few listings/property with same names has different hosts in different areas/neighbourhoods of a neighbourhood_group
df[df["host_name"] == "David"]["listing_name"].nunique()

# so here same host David operates different 402 listing/property

df[df["listing_name"] == df["host_name"]].head()

# there are few listings where the listing/property name and the host have same names
df.loc[(df["neighbourhood_group"] == "Queens") & (df["host_name"] == "Alex")].head(4)

# Same host have hosted different listing/property in different or same neighbourhood in same neighbourhood groups
# like Alex hosted different listings in most of different neighbourhood and there are same also in queens neighbourhood_group!


# Describe the Dataset and removing outliers
# describe the DataFrame
df.describe()

# Note - price column is very important so we have to find big outliers in important columns first.
sns.boxplot(x=df["price"])
plt.show()

### using IQR technique
# writing a outlier function for removing outliers in important columns.
def iqr_technique(DFcolumn):
    Q1 = np.percentile(DFcolumn, 25)
    Q3 = np.percentile(DFcolumn, 75)
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)  # interquantile range

    return lower_range, upper_range


lower_bound, upper_bound = iqr_technique(df["price"])

df = df[(df.price > lower_bound) & (df.price < upper_bound)]

# so the outliers are removed from price column now check with boxplot and also check shape of new Dataframe!

sns.boxplot(x=df["price"])
print(df.shape)

# so the outliers are removed from price column now check with boxplot and also check shape of new Dataframe!

sns.boxplot(x=df["price"])
print(df.shape)

# so here outliers are removed, see the new max price
print(df["price"].max())


# Data Visualization

#  Distribution Of Airbnb Bookings Price Range Using Histogram
# Create a figure with a custom size
plt.figure(figsize=(12, 5))

# Set the seaborn theme to darkgrid
sns.set_theme(style="darkgrid")

# Create a histogram of the 'price' column of the df dataframe
# using sns distplot function and specifying the color as red
sns.distplot(df["price"], color=("r"))

# Add labels to the x-axis and y-axis
plt.xlabel("Price", fontsize=14)
plt.ylabel("Density", fontsize=14)

# Add a title to the plot
plt.title("Distribution of Airbnb Prices", fontsize=15)


# Total Listing/Property count in Each Neighborhood Group using Count plot
# Count the number of listings in each neighborhood group and store the result in a Pandas series
# counts = df['neighbourhood_group'].value_counts()

# # Reset the index of the series so that the neighborhood groups become columns in the resulting dataframe
# Top_Neighborhood_group = counts.reset_index()

# # Rename the columns of the dataframe to be more descriptive
# Top_Neighborhood_group.columns = ['Neighborhood_Groups', 'Listing_Counts']

# # display the resulting DataFrame
# Top_Neighborhood_group

# # Set the figure size
# plt.figure(figsize=(12, 8))

# # Create a countplot of the neighbourhood group data
# sns.countplot(df['neighbourhood_group'])

# # Set the title of the plot
# plt.title('Neighbourhood_group Listing Counts in NYC', fontsize=15)

# # Set the x-axis label
# plt.xlabel('Neighbourhood_Group', fontsize=14)

# # Set the y-axis label
# plt.ylabel('total listings counts', fontsize=14)


# Average Price Of Each Neighborhood Group using Point Plot

# Group the Airbnb dataset by neighborhood group and calculate the mean of each group
grouped = df.groupby("neighbourhood_group").mean()

# Reset the index of the grouped dataframe so that the neighborhood group becomes a column
neighbourhood_group_avg_price = grouped.reset_index()

# Rename the "price" column to "avg_price"
neighbourhood_group_avg_price = round(
    neighbourhood_group_avg_price.rename(columns={"price": "avg_price"}), 2
)

# Select only the "neighbourhood_group" and "avg_price" columns
neighbourhood_group_avg_price[["neighbourhood_group", "avg_price"]].head()

# import mean function from the statistics module
from statistics import mean

# Create the point plot
sns.pointplot(x="neighbourhood_group", y="price", data=df, estimator=np.mean)

# Add axis labels and a title
plt.xlabel("Neighbourhood Group", fontsize=14)
plt.ylabel("Average Price", fontsize=14)
plt.title("Average Price by Neighbourhood Group", fontsize=15)

# Price Distribution Of Each Neighborhood Group using Violin Plot
# Create the violin plot for price distribution in each Neighbourhood_groups

ax = sns.violinplot(x="neighbourhood_group", y="price", data=df)


# Top Neighborhoods by Listing/property using Bar plot
# create a new DataFrame that displays the top 10 neighborhoods in the Airbnb NYC dataset based on the number of listings in each neighborhood
Top_Neighborhoods = df["neighbourhood"].value_counts()[:10].reset_index()

# rename the columns of the resulting DataFrame to 'Top_Neighborhoods' and 'Listing_Counts'
Top_Neighborhoods.columns = ["Top_Neighborhoods", "Listing_Counts"]

# display the resulting DataFrame
Top_Neighborhoods

# Get the top 10 neighborhoods by listing count
top_10_neigbourhoods = df["neighbourhood"].value_counts().nlargest(10)

# Create a list of colors to use for the bars
colors = [
    "c",
    "g",
    "olive",
    "y",
    "m",
    "orange",
    "#C0C0C0",
    "#800000",
    "#008000",
    "#000080",
]


# Create a bar plot of the top 10 neighborhoods using the specified colors
top_10_neigbourhoods.plot(kind="bar", figsize=(15, 6), color=colors)

# Set the x-axis label
plt.xlabel("Neighbourhood", fontsize=14)

# Set the y-axis label
plt.ylabel("Total Listing Counts", fontsize=14)

# Set the title of the plot
plt.title("Listings by Top Neighborhoods in NYC", fontsize=15)


# Average Minimum Price In Neighborhoods using Scatter and Bar chart

# create a new DataFrame that displays the average price of Airbnb rentals in each neighborhood
neighbourhood_avg_price = (
    df.groupby("neighbourhood")
    .mean()
    .reset_index()
    .rename(columns={"price": "avg_price"})[["neighbourhood", "avg_price"]]
)

# select the top 10 neighborhoods with the lowest average prices
neighbourhood_avg_price = neighbourhood_avg_price.sort_values("avg_price").head(10)

# join the resulting DataFrame with the 'neighbourhood_group' column from the Airbnb NYC dataset, dropping any duplicate entries
neighbourhood_avg_price_sorted_with_group = neighbourhood_avg_price.join(
    df[["neighbourhood", "neighbourhood_group"]]
    .drop_duplicates()
    .set_index("neighbourhood"),
    on="neighbourhood",
)


# Display the resulting data
display(neighbourhood_avg_price_sorted_with_group.style.hide_index())

neighbourhood_avg_price = (
    df.groupby("neighbourhood")
    .mean()
    .reset_index()
    .rename(columns={"price": "avg_price"})
)[["neighbourhood", "avg_price"]]
neighbourhood_avg_price = neighbourhood_avg_price.sort_values("avg_price")

# Group the data by neighborhood and calculate the average price
neighbourhood_avg_price = df.groupby("neighbourhood")["price"].mean()

# Create a new DataFrame with the average price for each neighborhood
neighbourhood_prices = pd.DataFrame(
    {
        "neighbourhood": neighbourhood_avg_price.index,
        "avg_price": neighbourhood_avg_price.values,
    }
)

# Merge the average price data with the original DataFrame#trying to find where the coordinates belong from the latitude and longitude
df = df.merge(neighbourhood_prices, on="neighbourhood")

# Create the scattermapbox plot
fig = df.plot.scatter(
    x="longitude",
    y="latitude",
    c="avg_price",
    title="Average Airbnb Price by Neighborhoods in New York City",
    figsize=(12, 6),
    cmap="plasma",
)
fig


#  Total Counts Of Each Room Type

# create a new DataFrame that displays the number of listings of each room type in the Airbnb NYC dataset
top_room_type = df["room_type"].value_counts().reset_index()

# rename the columns of the resulting DataFrame to 'Room_Type' and 'Total_counts'
top_room_type.columns = ["Room_Type", "Total_counts"]

# display the resulting DataFrame
top_room_type

# Set the figure size
plt.figure(figsize=(10, 6))

# Get the room type counts
room_type_counts = df["room_type"].value_counts()

# Set the labels and sizes for the pie chart
labels = room_type_counts.index
sizes = room_type_counts.values

# Create the pie chart
plt.pie(sizes, labels=labels, autopct="%1.1f%%")

# Add a legend to the chart
plt.legend(title="Room Type", bbox_to_anchor=(0.8, 0, 0.5, 1), fontsize="12")

# Show the plot
plt.show()


# Total Reviews by Each Neighborhood Group using Pie Chart

# Group the data by neighborhood group and calculate the total number of reviews
reviews_by_neighbourhood_group = df.groupby("neighbourhood_group")[
    "total_reviews"
].sum()

# Create a pie chart
plt.pie(
    reviews_by_neighbourhood_group,
    labels=reviews_by_neighbourhood_group.index,
    autopct="%1.1f%%",
)
plt.title("Number of Reviews by Neighborhood Group in New York City", fontsize="15")

# Display the chart
plt.show()


# most reviewed room type per month in neighbourhood groups
# create a figure with a default size of (10, 8)
f, ax = plt.subplots(figsize=(10, 8))

# create a stripplot that displays the number of reviews per month for each room type in the Airbnb NYC dataset
ax = sns.stripplot(
    x="room_type",
    y="reviews_per_month",
    hue="neighbourhood_group",
    dodge=True,
    data=df,
    palette="Set1",
)

# set the title of the plot
ax.set_title("Most Reviewed room_types in each Neighbourhood Groups", fontsize="14")

# Price variations in NYC Neighbourhood groups using scatter plot

# Let's have an idea of the price variations in neighborhood_groups

# create a scatter plot that displays the longitude and latitude of the listings in the Airbnb NYC dataset, with the color of each point indicating the price of the listing
lat_long = df.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    label="price_variations",
    c="price",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
    alpha=0.4,
    figsize=(10, 8),
)

# add a legend to the plot
lat_long.legend()


# Find Best Location Listing/Property Location For Travelers and Hosta

# Group the data by neighborhood and calculate the average number of reviews
neighbourhood_avg_reviews = df.groupby("neighbourhood")["total_reviews"].mean()

# Create a new DataFrame with the average number of reviews for each neighborhood
neighbourhood_reviews = pd.DataFrame(
    {
        "neighbourhood": neighbourhood_avg_reviews.index,
        "avg_reviews": neighbourhood_avg_reviews.values,
    }
)

# Merge the average number of reviews data with the original DataFrame
df = df.merge(neighbourhood_reviews, on="neighbourhood")

# Create the scattermapbox plot
fig = df.plot.scatter(
    x="longitude",
    y="latitude",
    c="avg_reviews",
    title="Average Airbnb Reviews by Neighborhoods in New York City",
    figsize=(14, 8),
    cmap="plasma",
)

# Display the scatter map
fig


#  Correlation Heatmap Visualization
# Calculate pairwise correlations between columns
corr = df.corr()

# Display the correlation between columns
corr

# Set the figure size
plt.figure(figsize=(12, 6))

# Visualize correlations as a heatmap
sns.heatmap(corr, cmap="BrBG", annot=True)

# Display heatmap
plt.show()
