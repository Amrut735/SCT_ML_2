import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class DataLoader:
    def __init__(self):
        self.scaler = StandardScaler()
        self.data = None
        self.scaled_data = None
        
    def load_sample_data(self):
        """Load sample Mall Customer Dataset"""
        # Create sample mall customer data
        np.random.seed(42)
        n_samples = 200
        
        # Generate realistic customer data
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
        
        self.data = data
        return data
    
    def load_csv_data(self, file_path):
        """Load data from uploaded CSV file"""
        try:
            data = pd.read_csv(file_path)
            self.data = data
            return data
        except Exception as e:
            raise Exception(f"Error loading CSV file: {str(e)}")
    
    def preprocess_data(self, features=None):
        """Preprocess data for clustering"""
        if self.data is None:
            raise Exception("No data loaded. Please load data first.")
        
        # Default features for customer segmentation
        if features is None:
            features = ['Age', 'Annual_Income_k', 'Spending_Score']
        
        # Check if required features exist
        missing_features = [f for f in features if f not in self.data.columns]
        if missing_features:
            raise Exception(f"Missing features in dataset: {missing_features}")
        
        # Select features and handle missing values
        feature_data = self.data[features].copy()
        feature_data = feature_data.dropna()
        
        if len(feature_data) == 0:
            raise Exception("No valid data after removing missing values")
        
        # Scale the features
        self.scaled_data = self.scaler.fit_transform(feature_data)
        
        return {
            'original_data': feature_data,
            'scaled_data': self.scaled_data,
            'feature_names': features
        }
    
    def get_data_summary(self):
        """Get summary statistics of the data"""
        if self.data is None:
            return None
        
        # Convert pandas data types to native Python types for JSON serialization
        summary = {
            'total_customers': int(len(self.data)),
            'features': [str(f) for f in list(self.data.columns)],
            'missing_values': {str(k): int(v) for k, v in self.data.isnull().sum().to_dict().items()},
            'data_types': {str(k): str(v) for k, v in self.data.dtypes.to_dict().items()}
        }
        
        # Add descriptive statistics for numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            desc_stats = self.data[numeric_cols].describe()
            # Convert all values to native Python types
            summary['descriptive_stats'] = {}
            for col in desc_stats.columns:
                summary['descriptive_stats'][str(col)] = {
                    str(stat): float(val) if not np.isnan(val) else None 
                    for stat, val in desc_stats[col].to_dict().items()
                }
        
        return summary 