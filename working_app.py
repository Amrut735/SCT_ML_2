from flask import Flask, render_template, request, jsonify, session
import os
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = 'customer_segmentation_key_2024'

# Create upload folder if it doesn't exist
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/test')
def test():
    """Test route"""
    return jsonify({"message": "Flask is working!", "status": "success"})

@app.route('/use_sample_data', methods=['POST'])
def use_sample_data():
    """Load and use sample data"""
    try:
        # Create sample data
        np.random.seed(42)
        n_samples = 200
        
        ages = np.random.normal(38, 13, n_samples).astype(int)
        ages = np.clip(ages, 18, 70)
        
        annual_income = np.random.normal(60, 26, n_samples).astype(int)
        annual_income = np.clip(annual_income, 15, 137)
        
        spending_score = np.random.normal(50, 25, n_samples).astype(int)
        spending_score = np.clip(spending_score, 1, 99)
        
        # Create DataFrame
        data = pd.DataFrame({
            'CustomerID': range(1, n_samples + 1),
            'Age': ages,
            'Annual_Income_k': annual_income,
            'Spending_Score': spending_score
        })
        
        # Create summary
        summary = {
            'total_customers': int(len(data)),
            'features': [str(f) for f in list(data.columns)],
            'missing_values': {str(k): int(v) for k, v in data.isnull().sum().to_dict().items()},
            'data_types': {str(k): str(v) for k, v in data.dtypes.to_dict().items()}
        }
        
        # Store in session
        session['data_summary'] = summary
        session['sample_data'] = True
        
        return jsonify({
            'success': True,
            'message': 'Sample data loaded successfully',
            'data_summary': summary
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
        
        # For now, just return a simple response
        return jsonify({
            'success': True,
            'message': 'Analysis endpoint is working',
            'optimal_k': 3,
            'used_k': 3,
            'n_clusters': 3
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003) 