import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

class ClusterVisualizer:
    def __init__(self):
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
    def create_cluster_scatter_plot(self, data, labels, feature_names, title="Customer Segmentation Clusters"):
        """Create 2D scatter plot of clusters"""
        plt.figure(figsize=(12, 8))
        
        # Use first two features for 2D visualization
        if len(feature_names) >= 2:
            x_feature, y_feature = feature_names[0], feature_names[1]
            x_data, y_data = data[:, 0], data[:, 1]
        else:
            # If less than 2 features, use PCA to reduce to 2D
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data)
            x_data, y_data = data_2d[:, 0], data_2d[:, 1]
            x_feature, y_feature = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
        
        # Create scatter plot
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(x_data[mask], y_data[mask], 
                       c=self.colors[i % len(self.colors)], 
                       label=f'Cluster {label}', 
                       alpha=0.7, s=60, edgecolors='white', linewidth=1)
        
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    
    def create_3d_scatter_plot(self, data, labels, feature_names):
        """Create 3D scatter plot of clusters"""
        if len(feature_names) < 3:
            return None
        
        fig = go.Figure()
        
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            fig.add_trace(go.Scatter3d(
                x=data[mask, 0],
                y=data[mask, 1],
                z=data[mask, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.colors[i % len(self.colors)],
                    opacity=0.8
                ),
                name=f'Cluster {label}'
            ))
        
        fig.update_layout(
            title="3D Customer Segmentation",
            scene=dict(
                xaxis_title=feature_names[0],
                yaxis_title=feature_names[1],
                zaxis_title=feature_names[2]
            ),
            width=800,
            height=600
        )
        
        return fig.to_html(include_plotlyjs='cdn', full_html=False)
    
    def create_feature_distribution_plots(self, data, labels, feature_names):
        """Create distribution plots for each feature by cluster"""
        n_features = len(feature_names)
        fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 5))
        
        if n_features == 1:
            axes = [axes]
        
        unique_labels = np.unique(labels)
        
        for i, feature in enumerate(feature_names):
            for j, label in enumerate(unique_labels):
                mask = labels == label
                axes[i].hist(data[mask, i], alpha=0.7, 
                           color=self.colors[j % len(self.colors)],
                           label=f'Cluster {label}', bins=20)
            
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    
    def create_cluster_heatmap(self, cluster_stats, feature_names):
        """Create heatmap showing cluster characteristics"""
        # Prepare data for heatmap
        heatmap_data = []
        cluster_names = []
        
        for cluster_id, stats in cluster_stats.items():
            cluster_names.append(f'Cluster {cluster_id}')
            row = [stats[feature]['mean'] for feature in feature_names]
            heatmap_data.append(row)
        
        heatmap_data = np.array(heatmap_data)
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data.T, 
                   annot=True, 
                   fmt='.1f',
                   xticklabels=cluster_names,
                   yticklabels=feature_names,
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Mean Value'})
        
        plt.title('Cluster Characteristics Heatmap')
        plt.xlabel('Clusters')
        plt.ylabel('Features')
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    
    def create_cluster_size_chart(self, labels):
        """Create bar chart showing cluster sizes"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(unique_labels, counts, 
                      color=[self.colors[i % len(self.colors)] for i in range(len(unique_labels))],
                      alpha=0.8)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Cluster')
        plt.ylabel('Number of Customers')
        plt.title('Cluster Size Distribution')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    
    def create_interactive_scatter(self, data, labels, feature_names):
        """Create interactive scatter plot using Plotly"""
        if len(feature_names) >= 2:
            x_feature, y_feature = feature_names[0], feature_names[1]
            x_data, y_data = data[:, 0], data[:, 1]
        else:
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data)
            x_data, y_data = data_2d[:, 0], data_2d[:, 1]
            x_feature, y_feature = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
        
        # Create DataFrame for Plotly
        df_plot = pd.DataFrame({
            'x': x_data,
            'y': y_data,
            'Cluster': [f'Cluster {label}' for label in labels]
        })
        
        fig = px.scatter(df_plot, x='x', y='y', color='Cluster',
                        title="Interactive Customer Segmentation",
                        labels={'x': x_feature, 'y': y_feature},
                        color_discrete_sequence=self.colors)
        
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        fig.update_layout(width=800, height=600)
        
        return fig.to_html(include_plotlyjs='cdn', full_html=False) 