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
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and validate data
            try:
                data = data_loader.load_csv_data(filepath)
                data_summary = data_loader.get_data_summary()
                
                # Store data info in session
                session['data_summary'] = data_summary
                session['filepath'] = filepath
                
                return jsonify({
                    'success': True,
                    'message': 'File uploaded successfully',
                    'data_summary': data_summary
                })
                
            except Exception as e:
                # Clean up file on error
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': str(e)}), 400
        
        return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

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
        
        # Generate cluster descriptions
        cluster_descriptions = analytics.generate_cluster_descriptions(
            cluster_stats,
            preprocessed['feature_names']
        )
        
        # Calculate business metrics
        business_metrics = analytics.calculate_business_metrics(
            cluster_stats,
            clustering_result['labels']
        )
        
        # Generate recommendations
        recommendations = analytics.generate_recommendations(
            cluster_descriptions,
            business_metrics
        )
        
        # Create visualizations
        elbow_plot = clustering.create_elbow_plot()
        scatter_plot = visualizer.create_cluster_scatter_plot(
            preprocessed['scaled_data'],
            clustering_result['labels'],
            preprocessed['feature_names']
        )
        distribution_plot = visualizer.create_feature_distribution_plots(
            preprocessed['scaled_data'],
            clustering_result['labels'],
            preprocessed['feature_names']
        )
        heatmap = visualizer.create_cluster_heatmap(
            cluster_stats,
            preprocessed['feature_names']
        )
        size_chart = visualizer.create_cluster_size_chart(
            clustering_result['labels']
        )
        
        # Create summary report
        summary_report = analytics.create_summary_report(
            cluster_stats,
            cluster_descriptions,
            business_metrics
        )
        
        # Store results in session
        session['analysis_results'] = {
            'optimal_k': optimal_k,
            'used_k': k_to_use,
            'cluster_stats': cluster_stats,
            'cluster_descriptions': cluster_descriptions,
            'business_metrics': business_metrics,
            'recommendations': recommendations,
            'feature_names': preprocessed['feature_names'],
            'cluster_labels': clustering_result['labels'].tolist()
        }
        
        return jsonify({
            'success': True,
            'optimal_k': optimal_k,
            'used_k': k_to_use,
            'cluster_stats': cluster_stats,
            'cluster_descriptions': cluster_descriptions,
            'business_metrics': business_metrics,
            'recommendations': recommendations,
            'summary_report': summary_report,
            'visualizations': {
                'elbow_plot': elbow_plot,
                'scatter_plot': scatter_plot,
                'distribution_plot': distribution_plot,
                'heatmap': heatmap,
                'size_chart': size_chart
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/download_results', methods=['POST'])
def download_results():
    """Download clustered data as CSV"""
    try:
        if 'analysis_results' not in session:
            return jsonify({'error': 'No analysis results available'}), 400
        
        # Get original data
        if session.get('sample_data'):
            data = data_loader.load_sample_data()
        else:
            filepath = session.get('filepath')
            data = data_loader.load_csv_data(filepath)
        
        # Add cluster labels
        results = session['analysis_results']
        data['Cluster'] = results['cluster_labels']
        
        # Create CSV in memory
        output = io.StringIO()
        data.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='customer_segmentation_results.csv'
        )
        
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/get_data_preview', methods=['GET'])
def get_data_preview():
    """Get preview of loaded data"""
    try:
        if 'data_summary' not in session:
            return jsonify({'error': 'No data available'}), 400
        
        # Load data
        if session.get('sample_data'):
            data = data_loader.load_sample_data()
        else:
            filepath = session.get('filepath')
            data = data_loader.load_csv_data(filepath)
        
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

@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear session data"""
    try:
        # Clean up uploaded files
        if 'filepath' in session:
            filepath = session['filepath']
            if os.path.exists(filepath):
                os.remove(filepath)
        
        # Clear session
        session.clear()
        
        return jsonify({'success': True, 'message': 'Session cleared successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Failed to clear session: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 