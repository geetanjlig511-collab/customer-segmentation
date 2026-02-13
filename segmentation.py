import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score   # Added

# Load CSV
df = pd.read_csv("customer_data.csv")
print("CSV Loaded Successfully!\n")
print(df.head())

# Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Reference date for Recency
max_date = df["InvoiceDate"].max()

# RFM Calculation
rfm = df.groupby("CustomerID").agg(
    Recency = ("InvoiceDate", lambda x: (max_date - x.max()).days),
    Frequency = ("InvoiceDate", "count"),
    Monetary = ("Amount", "sum")
)

print("\nRFM Table:\n", rfm.head())

# Scaling for clustering
scaler = StandardScaler()
scaled = scaler.fit_transform(rfm)

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(scaled)

print("\nClustering Done!\n", rfm.head())

# Silhouette Score (Accuracy Measure)
score = silhouette_score(scaled, rfm["Cluster"])
print("\nSilhouette Score:", score)

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(rfm["Recency"], rfm["Monetary"], c=rfm["Cluster"], cmap='viridis')
plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.title("RFM Customer Segmentation")
plt.show()

# Save output CSV
rfm.to_csv("segmentation.csv")
print("\nOutput CSV saved as 'segmentation.csv'")
