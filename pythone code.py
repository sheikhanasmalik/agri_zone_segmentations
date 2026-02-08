# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 16:23:34 2026

@author: Samsung
"""

# ---------------------------------------
# 1. Import Required Libraries
# ---------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------
# 2. Load Dataset
# ---------------------------------------
df = pd.read_csv(r"C:\Anas\Data Science\Agri Zone Segmentation\agriculture_field_data_500_rows.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ---------------------------------------
# 3. Drop Non-Numeric / Identifier Columns
# ---------------------------------------
features = ["rainfall_mm","temperature_c",]
X = df[features]


# ---------------------------------------
# 4. Feature Scaling (VERY IMPORTANT)
# ---------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------
# 5. Elbow Method to Find Optimal K
# ---------------------------------------
wcss = []

K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(K_range, wcss, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal k")
plt.show()

# ---------------------------------------
# 6. Silhouette Score for Different K
# ---------------------------------------
for k in K_range:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score for k={k}: {score:.4f}")

# ---------------------------------------
# 7. Final K-Means Model (Choose Best k)
# ---------------------------------------
kmeans = KMeans(n_clusters=4, random_state=42)



# Fit the model and predict cluster labels

clusters = kmeans.fit_predict(X_scaled)



# Add cluster labels to original dataframe

df['Cluster'] = clusters



print(df.head())





# -------- Step 7: Evaluate clustering using Silhouette Score --------

# Silhouette score measures how well points fit within their cluster

sil_score = silhouette_score(X_scaled, clusters)



print("Silhouette Score:", sil_score)





# -------- Step 8: Visualize the clusters --------

plt.figure(figsize=(8, 6))



plt.scatter(

    X.iloc[:, 0], # Annual Income

    X.iloc[:, 1], # Spending Score

    c=clusters # Cluster labels as colors

)



plt.xlabel("Rainfall in mm (60-150)")

plt.ylabel("Temperature in C (25-36)")

plt.title("Agri Zone Segmentation using K-Means")

plt.show()