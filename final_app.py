from flask import Flask, render_template, request, jsonify, send_file, session
import os
import pandas as pd
import numpy as np
import json
from werkzeug.utils import secure_filename
import tempfile
import io
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import base64

app = Flask(__name__)
app.secret_key = 'customer_segmentation_key_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_sample_data():
    """Load sample Mall Customer Dataset"""
    np.random.seed(42)
    n_samples = 200
    
    ages = np.random.normal(38, 13, n_samples).astype(int)
    ages = np.clip(ages, 18, 70)
    
    annual_income = np.random.normal(60, 26, n_samples).astype(int)
    annual_income = np.clip(annual_income, 15, 137)
    
    spending_score = np.random.normal(50, 25, n_samples).astype(int)
    spending_score = np.clip(spending_score, 1, 99)
    
    data = pd.DataFrame({
        'CustomerID': range(1, n_samples + 1),
        'Age': ages,
        'Annual_Income_k': annual_income,
        'Spending_Score': spending_score
    })
    
    return data

def get_data_summary(data):
    """Get summary statistics of the data"""
    summary = {
        'total_customers': int(len(data)),
        'features': [str(f) for f in list(data.columns)],
        'missing_values': {str(k): int(v) for k, v in data.isnull().sum().to_dict().items()},
        'data_types': {str(k): str(v) for k, v in data.dtypes.to_dict().items()}
    }
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        desc_stats = data[numeric_cols].describe()
        summary['descriptive_stats'] = {}
        for col in desc_stats.columns:
            summary['descriptive_stats'][str(col)] = {
                str(stat): float(val) if not np.isnan(val) else None 
                for stat, val in desc_stats[col].to_dict().items()
            }
    
    return summary

def find_optimal_k(data, max_k=10):
    """Find optimal number of clusters using Elbow method and Silhouette analysis"""
    if len(data) < 10:
        return min(3, len(data) - 1)
    
    # Constrain k based on data size
    max_k = min(max_k, min(5, len(data) // 10))
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        
        if k > 1:
            labels = kmeans.labels_
            silhouette_avg = silhouette_score(data, labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(0)
    
    # Elbow method
    elbow_k = 3  # Default
    if len(inertias) > 2:
        # Simple elbow detection
        for i in range(1, len(inertias) - 1):
            if inertias[i-1] - inertias[i] > inertias[i] - inertias[i+1]:
                elbow_k = k_range[i]
                break
    
    # Silhouette method
    silhouette_k = k_range[np.argmax(silhouette_scores)]
    
    # Combine both methods
    optimal_k = min(elbow_k, silhouette_k)
    
    return optimal_k

def perform_clustering(data, k):
    """Perform KMeans clustering"""
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
    labels = kmeans.fit_predict(data)
    
    return {
        'labels': labels,
        'centroids': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_
    }

def calculate_cluster_statistics(data, labels, feature_names):
    """Calculate statistics for each cluster"""
    cluster_stats = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        mask = labels == label
        cluster_data = data[mask]
        
        stats = {}
        for i, feature in enumerate(feature_names):
            feature_data = cluster_data[:, i]
            stats[str(feature)] = {
                'mean': float(np.mean(feature_data)),
                'median': float(np.median(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'count': int(len(feature_data))
            }
        
        cluster_stats[int(label)] = stats
    
    return cluster_stats

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/use_sample_data', methods=['POST'])
def use_sample_data():
    """Load and use sample data"""
    try:
        # Load sample data
        data = load_sample_data()
        data_summary = get_data_summary(data)
        
        # Store in session
        session['data_summary'] = data_summary
        session['sample_data'] = True
        
        return jsonify({
            'success': True,
            'message': 'Sample data loaded successfully',
            'data_summary': data_summary
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load sample data: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """Perform clustering analysis"""
    try:
        # Check if data is available
        if 'data_summary' not in session:
            return jsonify({'error': 'No data available. Please upload or load sample data first.'}), 400
        
        # Get custom parameters
        custom_k = request.json.get('custom_k', None)
        features = request.json.get('features', ['Age', 'Annual_Income_k', 'Spending_Score'])
        
        # Load data
        if session.get('sample_data'):
            data = load_sample_data()
        else:
            filepath = session.get('filepath')
            if not filepath or not os.path.exists(filepath):
                return jsonify({'error': 'Data file not found. Please upload again.'}), 400
            data = pd.read_csv(filepath)
        
        # Select features
        feature_data = data[features].dropna()
        
        # Scale the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)
        
        # Find optimal k
        optimal_k = find_optimal_k(scaled_data)
        
        # Perform clustering
        k_to_use = custom_k if custom_k else optimal_k
        clustering_result = perform_clustering(scaled_data, k_to_use)
        
        # Calculate cluster statistics
        cluster_stats = calculate_cluster_statistics(
            feature_data.values,
            clustering_result['labels'],
            features
        )
        
        # Store results in session
        session['analysis_results'] = {
            'optimal_k': optimal_k,
            'used_k': k_to_use,
            'cluster_stats': cluster_stats,
            'feature_names': features,
            'cluster_labels': clustering_result['labels'].tolist()
        }
        
        return jsonify({
            'success': True,
            'optimal_k': optimal_k,
            'used_k': k_to_use,
            'cluster_stats': cluster_stats,
            'n_clusters': len(cluster_stats)
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/get_data_preview', methods=['GET'])
def get_data_preview():
    """Get preview of loaded data"""
    try:
        if 'data_summary' not in session:
            return jsonify({'error': 'No data available'}), 400
        
        # Load data
        if session.get('sample_data'):
            data = load_sample_data()
        else:
            filepath = session.get('filepath')
            data = pd.read_csv(filepath)
        
        # Return first 10 rows
        preview = data.head(10).to_dict('records')
        columns = list(data.columns)
        
        return jsonify({
            'preview': preview,
            'columns': columns,
            'total_rows': len(data)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get data preview: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 