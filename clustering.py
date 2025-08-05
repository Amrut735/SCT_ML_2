import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import io
import base64

class CustomerSegmentation:
    def __init__(self):
        self.kmeans_model = None
        self.optimal_k = None
        self.cluster_labels = None
        self.elbow_data = None
        
    def find_optimal_k(self, data, max_k=10):
        """Find optimal number of clusters using Elbow method and Silhouette analysis"""
        if len(data) < 2:
            raise Exception("Insufficient data for clustering")
        
        # Limit max_k based on data size and constraints
        max_k = min(max_k, len(data) // 10, 5)  # Ensure at least 10 samples per cluster, max 5 clusters
        
        if max_k < 2:
            max_k = 2
        
        k_range = range(2, max_k + 1)
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            # Fit KMeans
            kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
            kmeans.fit(data)
            
            # Calculate inertia (Elbow method)
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            if k > 1:  # Silhouette score requires at least 2 clusters
                labels = kmeans.labels_
                silhouette_avg = silhouette_score(data, labels)
                silhouette_scores.append(silhouette_avg)
            else:
                silhouette_scores.append(0)
        
        # Find optimal k using elbow method
        # Calculate the rate of change of inertia
        inertia_changes = np.diff(inertias)
        inertia_change_rates = np.abs(np.diff(inertia_changes))
        
        # Find the elbow point (where the rate of change starts to level off)
        if len(inertia_change_rates) > 0:
            # Find the point where the change rate drops significantly
            threshold = np.mean(inertia_change_rates) * 0.5
            elbow_k = None
            for i, rate in enumerate(inertia_change_rates):
                if rate < threshold:
                    elbow_k = k_range[i + 1]
                    break
            
            if elbow_k is None:
                elbow_k = k_range[np.argmin(inertia_change_rates)]
        else:
            elbow_k = 2
        
        # Also consider silhouette score
        if len(silhouette_scores) > 0:
            silhouette_k = k_range[np.argmax(silhouette_scores)]
            # Use the average of elbow and silhouette methods
            self.optimal_k = int(np.round((elbow_k + silhouette_k) / 2))
        else:
            self.optimal_k = elbow_k
        
        # Ensure optimal_k is within bounds
        self.optimal_k = max(2, min(self.optimal_k, max_k))
        
        # Store elbow data for plotting
        self.elbow_data = {
            'k_values': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': self.optimal_k
        }
        
        return self.optimal_k
    
    def perform_clustering(self, data, k=None):
        """Perform KMeans clustering on the data"""
        if k is None:
            k = self.optimal_k or self.find_optimal_k(data)
        
        # Ensure k is within constraints
        k = min(k, 5, len(data) // 10)  # Max 5 clusters, at least 10 samples per cluster
        k = max(2, k)  # At least 2 clusters
        
        # Fit KMeans model
        self.kmeans_model = KMeans(
            n_clusters=k,
            n_init=10,
            max_iter=300,
            random_state=42
        )
        
        self.cluster_labels = self.kmeans_model.fit_predict(data)
        
        return {
            'model': self.kmeans_model,
            'labels': self.cluster_labels,
            'centroids': self.kmeans_model.cluster_centers_,
            'inertia': self.kmeans_model.inertia_,
            'n_clusters': k
        }
    
    def get_cluster_centers(self):
        """Get cluster centers in original scale"""
        if self.kmeans_model is None:
            return None
        return self.kmeans_model.cluster_centers_
    
    def predict_cluster(self, data):
        """Predict cluster for new data"""
        if self.kmeans_model is None:
            raise Exception("Model not fitted. Please perform clustering first.")
        return self.kmeans_model.predict(data)
    
    def create_elbow_plot(self):
        """Create elbow plot for optimal k selection"""
        if self.elbow_data is None:
            return None
        
        plt.figure(figsize=(12, 5))
        
        # Elbow plot
        plt.subplot(1, 2, 1)
        plt.plot(self.elbow_data['k_values'], self.elbow_data['inertias'], 'bo-')
        plt.axvline(x=self.elbow_data['optimal_k'], color='red', linestyle='--', 
                   label=f'Optimal k = {self.elbow_data["optimal_k"]}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Silhouette plot
        plt.subplot(1, 2, 2)
        plt.plot(self.elbow_data['k_values'], self.elbow_data['silhouette_scores'], 'go-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    
    def get_clustering_summary(self):
        """Get summary of clustering results"""
        if self.kmeans_model is None:
            return None
        
        return {
            'n_clusters': self.kmeans_model.n_clusters,
            'inertia': self.kmeans_model.inertia_,
            'n_iterations': self.kmeans_model.n_iter_,
            'converged': self.kmeans_model.n_iter_ < 300
        } 