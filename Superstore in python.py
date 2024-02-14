import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset from the GitHub Gist link
url = "https://gist.githubusercontent.com/John-Ted-07/f197722b69d52a6dbb70f33ba2b7530a/raw/8b4df4c845af12c42d4f11a8e79371ff54d71754/sales_data.csv"
df = pd.read_csv(url)

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Check the data types of each column
print(df.dtypes)

# Check for missing values
print(df.isnull().sum())

# Summary statistics of numerical columns
print(df.describe())

# Summary statistics of categorical columns
print(df.describe(include=['object']))

# Unique values in categorical columns
print(df['Product_ID'].unique())
print(df['Region'].unique())
print(df['Order_Priority'].unique())

# Frequency count of categorical variables
print(df['Region'].value_counts())
print(df['Order_Priority'].value_counts())

# Correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)

# Visualize correlation matrix (optional)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Boxplot for numerical variables
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_columns])
plt.title('Boxplot of Numerical Variables')
plt.xticks(rotation=45)
plt.show()

# Histogram for numerical variables
plt.figure(figsize=(12, 6))
df[numerical_columns].hist(bins=20, figsize=(12, 6))
plt.suptitle('Histogram of Numerical Variables', y=1.02)
plt.show()

# Bar plot for categorical variables
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=col, data=df)
    plt.title(f'Bar Plot of {col}')
    plt.xticks(rotation=45)
    plt.show()

# Scatter plot between numerical variables
sns.pairplot(df[numerical_columns])
plt.suptitle('Pairplot of Numerical Variables', y=1.02)
plt.show()

# Grouping and aggregation
grouped_data = df.groupby('Region')['Sales'].sum().reset_index()
print(grouped_data)

# Sorting
sorted_data = df.sort_values(by='Sales', ascending=False)
print(sorted_data.head())

# Filtering
filtered_data = df[df['Sales'] > 10000]
print(filtered_data.head())

# Creating new features (if applicable)
df['Revenue'] = df['Sales'] * df['Quantity']

# Data transformation (if needed)
df['Log_Sales'] = np.log(df['Sales'])

# Time series analysis (if applicable)
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Year'] = df['Order_Date'].dt.year
df['Month'] = df['Order_Date'].dt.month
df['Day'] = df['Order_Date'].dt.day

# Group by year and visualize sales trend over time
yearly_sales = df.groupby('Year')['Sales'].sum()
plt.figure(figsize=(10, 6))
yearly_sales.plot(kind='line', marker='o')
plt.title('Yearly Sales Trend')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()

# Customer segmentation (if applicable)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Sales', 'Quantity']])

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Sales', y='Quantity', hue='Cluster', data=df, palette='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Sales')
plt.ylabel('Quantity')
plt.show()

cluster_means = df.groupby('Cluster')[['Sales', 'Quantity']].mean()
print(cluster_means)

# Export the dataset with cluster labels
df.to_csv('clustered_sales_data.csv', index=False)

