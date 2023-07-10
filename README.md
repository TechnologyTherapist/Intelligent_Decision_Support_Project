# Intelligent Decision Support Project

This project aims to provide intelligent decision support using data analysis and visualization techniques on the Airbnb NYC 2019 dataset. The project utilizes Python libraries such as NumPy, Pandas, Matplotlib, and Seaborn for data processing, exploration, and visualization.

## Dataset

The project uses the Airbnb NYC 2019 dataset, which contains information about various Airbnb listings in New York City. The dataset includes details like listing IDs, names, neighborhood groups, neighborhoods, room types, prices, host information, availability, and reviews.

## Data Exploration and Cleaning

The initial phase involves exploring the dataset, checking variable information, and identifying missing values and duplicates. The dataset columns are renamed for better understanding. Null values are handled by filling them with appropriate values or dropping columns that are not essential for analysis.

## Data Analysis and Visualization

The project performs various data analysis and visualization tasks to gain insights into the dataset. Here are some of the analyses and visualizations performed:

1. Distribution of Airbnb Prices: A histogram is created to visualize the price range of Airbnb bookings.

2. Total Listing Counts in Each Neighborhood Group: The count of listings in each neighborhood group is represented using a bar plot.

3. Average Price of Each Neighborhood Group: The average price of Airbnb listings in each neighborhood group is visualized using a point plot.

4. Price Distribution of Each Neighborhood Group: Violin plots are used to show the price distribution in each neighborhood group.

5. Top Neighborhoods by Listing/Property: Bar plots display the top neighborhoods based on the number of listings.

6. Average Minimum Price in Neighborhoods: Scatter and bar charts present the average minimum prices in different neighborhoods.

7. Total Counts of Each Room Type: A pie chart illustrates the distribution of different room types.

8. Total Reviews by Each Neighborhood Group: A pie chart displays the distribution of reviews among neighborhood groups.

9. Most Reviewed Room Type per Month in Neighborhood Groups: A strip plot shows the number of reviews per month for each room type in different neighborhood groups.

10. Price Variations in NYC Neighborhood Groups: A scatter plot represents the price variations across different neighborhood groups using the longitude and latitude of listings.

11. Best Location for Travelers and Hosts: A scatter map showcases the average number of reviews for each neighborhood, helping identify the best locations.

12. Correlation Heatmap Visualization: A heatmap depicts the pairwise correlations between columns in the dataset.

These analyses and visualizations provide valuable insights into the Airbnb NYC dataset, helping users make informed decisions.

Please note that this is a summary of the project and the actual code contains more detailed implementation and additional visualizations.

For more details and the complete code, please refer to the [Jupyter Notebook](link-to-your-jupyter-notebook-file) in this repository.

## Acknowledgments

- Airbnb NYC 2019 dataset: [Kaggle](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)
- Python libraries: NumPy, Pandas, Matplotlib, Seaborn
