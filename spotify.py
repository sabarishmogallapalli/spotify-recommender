import pandas as pd
from sklearn import preprocessing, cluster, metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Load and clean the dataset
df = pd.read_csv('spotify_data.csv')
df = df[['title', 'artist', 'top genre', 'bpm', 'nrgy', 'dnce', 'dB', 'live', 'val', 'dur', 'acous', 'spch', 'pop']]

# Encode categorical features
label_encoder = preprocessing.LabelEncoder()
df['top genre'] = label_encoder.fit_transform(df['top genre'])

# Define feature subsets
mood_features = ['val', 'nrgy', 'acous', 'live']
popularity_features = ['pop', 'dur']
rhythm_features = ['bpm', 'dnce']
all_features = ['top genre', 'bpm', 'nrgy', 'dnce', 'dB', 'live', 'val', 'dur', 'acous', 'spch', 'pop']

# Experimenting with mood features
X = df[all_features]  # You can switch to other feature groups like rhythm_features or all_features

# Standardize the data
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=1)  # Retain 95% variance
X_pca = pca.fit_transform(X_scaled)

# Correlation heatmap of PCA components
X_pca_df = pd.DataFrame(X_pca, columns=['PCA1'])
corr = X_pca_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of PCA Components')
plt.show()

X = df[['top genre', 'bpm', 'nrgy', 'dnce', 'dB', 'live', 'val', 'dur', 'acous', 'spch', 'pop']]
corr2 = X.corr()
# Create the heatmap
sns.heatmap(corr2, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of All Features')
plt.show()

X = df[['val', 'nrgy', 'acous', 'live']]
corr2 = X.corr()
# Create the heatmap
sns.heatmap(corr2, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Mood Features')
plt.show()

X = df[['pop', 'dur']]
corr3 = X.corr()
# Create the heatmap
sns.heatmap(corr3, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Popularity Features')
plt.show()

X = df[['bpm', 'dnce']]
corr4 = X.corr()
# Create the heatmap
sns.heatmap(corr4, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Rhythm Features')
plt.show()

# Optimize the number of clusters
silhouette_scores = []
range_n_clusters = range(2, 5)

for n_clusters in range_n_clusters:
    kmeans_test = cluster.KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans_test.fit_predict(X_pca)
    silhouette = metrics.silhouette_score(X_pca, cluster_labels)
    silhouette_scores.append(silhouette)

# Plot silhouette scores to choose the best number of clusters
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title("Silhouette Score vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# Choose the optimal number of clusters
optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal Number of Clusters: {optimal_n_clusters}")

# Apply KMeans clustering on PCA-transformed data
kmeans = cluster.KMeans(n_clusters=optimal_n_clusters, random_state=42)
kmeans.fit(X_pca)

# Add the PCA-based clusters to the dataframe
df['mood_cluster_pca'] = kmeans.predict(X_pca)

# Evaluate clustering with Silhouette Score
silhouette_pca = metrics.silhouette_score(X_pca, df['mood_cluster_pca'])
print(f"Silhouette Score (PCA): {silhouette_pca:.2f}")

# Visualize the clusters
plt.scatter(X_pca[:, 0], [0] * len(X_pca), c=kmeans.labels_, cmap='viridis', s=50, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], [0] * len(kmeans.cluster_centers_), s=200, c='red', marker='X')
plt.title("Cluster Visualization")
plt.xlabel("PCA Component 1")
plt.yticks([])  # Hides y-axis since it's constant (0)
plt.show()

# Define a simulated user input for "Nostalgic Yet Optimistic" mood
user_input_nostalgic_optimistic = { #add more example maps for different words/phrases
    'top genre': label_encoder.transform(['pop'])[0],
    'bpm': 85,  # Moderate tempo
    'nrgy': 50,  # Balanced energy
    'dnce': 60,  # Rhythmic but not overwhelming
    'dB': -15,   # Soft volume
    'live': 40,  # Moderate live performance feel
    'val': 70,   # Optimistic and positive valence
    'dur': 240,  # Longer duration for storytelling songs
    'acous': 80, # High acousticness for natural sounds
    'spch': 10,  # Minimal speech
    'pop': 65    # Moderately popular
}

# Convert user input into a dataframe
user_input_df = pd.DataFrame([user_input_nostalgic_optimistic])

# Standardize the user input using the same scaler
user_input_scaled = scaler.transform(user_input_df)

# Apply PCA to user input
user_input_scaled_pca = pca.transform(user_input_scaled)

# Predict the cluster for the user's mood input
user_mood_cluster_pca = kmeans.predict(user_input_scaled_pca)[0]

# Filter songs from the predicted cluster
recommended_songs_pca = df[df['mood_cluster_pca'] == user_mood_cluster_pca]

# Prepare for KNN recommendation
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_pca, df['mood_cluster_pca'])

# Predict songs similar to user input within the cluster
distances, indices = knn.kneighbors(user_input_scaled_pca)
similar_songs_pca = df.iloc[indices[0]]

# Display results
print("Predicted Mood Cluster for the User Input (PCA):", user_mood_cluster_pca)
print("\nSongs in the Same Cluster (KMeans with PCA):")
print(recommended_songs_pca[['title', 'artist', 'mood_cluster_pca']])

print("\nSongs Recommended Based on User Input (KNN with PCA):")
print(similar_songs_pca[['title', 'artist', 'mood_cluster_pca']])