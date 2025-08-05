from flask import Flask, render_template, request, jsonify, send_file, session
import os
import pandas as pd
import numpy as np
import json
from werkzeug.utils import secure_filename
import tempfile
import io

from data_loader import DataLoader
from clustering import CustomerSegmentation
from visualize import ClusterVisualizer
from analytics import ClusterAnalytics

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

# Initialize components
data_loader = DataLoader()
clustering = CustomerSegmentation()
visualizer = ClusterVisualizer()
analytics = ClusterAnalytics()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error rendering template: {str(e)}", 500

@app.route('/test')
def test():
    """Test route"""
    return "Customer Segmentation App is working!"

@app.route('/use_sample_data', methods=['POST'])
def use_sample_data():
    """Load and use sample data"""
    try:
        # Load sample data
        data = data_loader.load_sample_data()
        data_summary = data_loader.get_data_summary()
        
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
            data = data_loader.load_sample_data()
        else:
            filepath = session.get('filepath')
            if not filepath or not os.path.exists(filepath):
                return jsonify({'error': 'Data file not found. Please upload again.'}), 400
            data = data_loader.load_csv_data(filepath)
        
        # Preprocess data
        preprocessed = data_loader.preprocess_data(features)
        
        # Find optimal k
        optimal_k = clustering.find_optimal_k(preprocessed['scaled_data'])
        
        # Perform clustering
        k_to_use = custom_k if custom_k else optimal_k
        clustering_result = clustering.perform_clustering(preprocessed['scaled_data'], k_to_use)
        
        # Calculate cluster statistics
        cluster_stats = analytics.calculate_cluster_statistics(
            preprocessed['original_data'].values,
            clustering_result['labels'],
            preprocessed['feature_names']
        )
        
        return jsonify({
            'success': True,
            'optimal_k': optimal_k,
            'used_k': k_to_use,
            'cluster_stats': cluster_stats,
            'n_clusters': len(cluster_stats)
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 