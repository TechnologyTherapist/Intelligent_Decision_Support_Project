import pandas as pd

# Load the Stores tab
stores_data = pd.read_csv('/Volumes/DATA/Data Analyst Projects/Intelligent_Decision_Support_Project/data/stores data-set.csv')

# Load the Features tab
features_data = pd.read_csv('/Volumes/DATA/Data Analyst Projects/Intelligent_Decision_Support_Project/data/Features data set.csv')

# Load the Sales tab
sales_data = pd.read_csv('/Volumes/DATA/Data Analyst Projects/Intelligent_Decision_Support_Project/data/sales data-set.csv')

# Print the first few rows of each dataset for verification
print("Stores Data:")
print(stores_data.head())

print("\nFeatures Data:")
print(features_data.head())

print("\nSales Data:")
print(sales_data.head())
