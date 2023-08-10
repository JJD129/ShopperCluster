# ShopperCluster

## Libraries Used
```
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
```
## Data Source
The dataset named "Customers.csv" is sourced from Kaggle and located in the /kaggle/input/customers-dataset/ directory.
https://www.kaggle.com/datasets/datascientistanna/customers-dataset

Shop Customer Data is a detailed analysis of an imaginative shop's ideal customers. It helps a business to better understand its customers. The owner of a shop gets information about Customers through membership cards.

The dataset consists of 2000 records and 8 columns:

* Customer ID
* Gender
* Age
* Annual Income
* Spending Score - Score assigned by the shop, based on customer behavior and spending nature
* Profession
* Work Experience - in years
* Family Size

## Data Exploration and Visualization
1. Loading Data: The dataset is loaded into a dataframe using pandas. Initial exploration includes checking the head of the dataframe, unique values, and missing values.
2. Visualizations: Various visualizations are generated to explore relationships in the data. These visualizations include:
* Pie charts to understand the distribution of average income by different categories (e.g., Gender or Profession).
* Box plots to observe the distribution of various numerical features across different categorical features.
* Scatter plots to visualize the relationships between numerical features.
3. Data Preprocessing: Steps here include:
* Addressing skewness in the data by applying transformations.
* Encoding categorical variables using LabelEncoder.
* Scaling the features for better clustering results using StandardScaler.

## Clustering with K-means
1. Determining the Number of Clusters:
* The Elbow Method and Silhouette Method are used to determine the optimal number of clusters for K-means clustering.
2. Visualizing Clusters: After fitting K-means with the optimal number of clusters, the clusters are visualized using t-SNE for dimensionality reduction.

## Classification using Random Forest
1. Data Splitting: The data is split into training and testing sets using a 80-20 split.
2. Model Training and Prediction: A Random Forest classifier is trained on the training set and then predictions are made on the testing set.
3. Evaluation: The performance of the Random Forest classifier is evaluated using accuracy, classification report, and a confusion matrix.
4. Feature Importance: The importance of each feature in making predictions is visualized using a bar chart.

**Note**
Remember to change the paths or configurations based on your setup if necessary. The tutorial assumes that the code is being executed in a Kaggle notebook environment.
