import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Data Import
df = pd.read_csv("/Volumes/DATA/Data Analyst Projects/Intelligent_Decision_Support_Project/data/AB_NYC_2019.csv")

# Data Cleaning
df.drop_duplicates(inplace=True)
df["listing_name"].fillna("unknown", inplace=True)
df["host_name"].fillna("no_name", inplace=True)
df.drop(["last_review"], axis=1, inplace=True)
df["reviews_per_month"].fillna(0, inplace=True)

# Data Preprocessing
le = LabelEncoder()
df["neighbourhood_group"] = le.fit_transform(df["neighbourhood_group"])
df["room_type"] = le.fit_transform(df["room_type"])

# Feature Selection
X = df.drop(["price"], axis=1)
y = df["price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Feature Importance
feature_importance = model.feature_importances_
sorted_indices = np.argsort(feature_importance)[::-1]
sorted_features = X.columns[sorted_indices]
sorted_importance = feature_importance[sorted_indices]

# Visualization: Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_importance, y=sorted_features)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Decision Making")
plt.show()

# Integration with Decision-Support Systems
# You can integrate the trained model or the identified important features with your decision-support systems.

# Further Analysis and Reporting
# Perform additional analysis as required by your project and generate comprehensive reports or presentations.

