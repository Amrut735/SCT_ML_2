#!/usr/bin/env python3
"""
Test script for Customer Segmentation Application
This script tests the core functionality without requiring a web browser.
"""

import requests
import json
import time

def test_application():
    """Test the main application endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Customer Segmentation Application...")
    print("=" * 50)
    
    # Test 1: Check if server is running
    print("1. Testing server connectivity...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ Server is running successfully!")
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    # Test 2: Test sample data loading
    print("\n2. Testing sample data loading...")
    try:
        response = requests.post(f"{base_url}/use_sample_data", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Sample data loaded successfully!")
                print(f"   - Total customers: {data['data_summary']['total_customers']}")
                print(f"   - Features: {len(data['data_summary']['features'])}")
            else:
                print(f"❌ Failed to load sample data: {data.get('error')}")
                return False
        else:
            print(f"❌ Sample data request failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Sample data request error: {e}")
        return False
    
    # Test 3: Test clustering analysis
    print("\n3. Testing clustering analysis...")
    try:
        analysis_params = {
            "features": ["Age", "Annual_Income_k", "Spending_Score"],
            "custom_k": None
        }
        response = requests.post(
            f"{base_url}/analyze",
            json=analysis_params,
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Clustering analysis completed successfully!")
                print(f"   - Optimal k: {data['optimal_k']}")
                print(f"   - Used k: {data['used_k']}")
                print(f"   - Number of clusters: {len(data['cluster_stats'])}")
                
                # Print cluster information
                for cluster_id, stats in data['cluster_stats'].items():
                    age_mean = stats['Age']['mean']
                    income_mean = stats['Annual_Income_k']['mean']
                    spending_mean = stats['Spending_Score']['mean']
                    count = stats['Age']['count']
                    print(f"   - Cluster {cluster_id}: {count} customers, "
                          f"Avg Age: {age_mean:.1f}, "
                          f"Avg Income: ${income_mean:.1f}k, "
                          f"Avg Spending: {spending_mean:.1f}")
            else:
                print(f"❌ Analysis failed: {data.get('error')}")
                return False
        else:
            print(f"❌ Analysis request failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Analysis request error: {e}")
        return False
    
    # Test 4: Test data preview
    print("\n4. Testing data preview...")
    try:
        response = requests.get(f"{base_url}/get_data_preview", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'preview' in data:
                print("✅ Data preview retrieved successfully!")
                print(f"   - Preview rows: {len(data['preview'])}")
                print(f"   - Total rows: {data['total_rows']}")
                print(f"   - Columns: {data['columns']}")
            else:
                print("❌ Data preview failed")
                return False
        else:
            print(f"❌ Data preview request failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Data preview request error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Application is working correctly.")
    print(f"🌐 Open your browser and go to: {base_url}")
    print("📊 Try the sample data or upload your own CSV file!")
    return True

if __name__ == "__main__":
    # Wait a moment for the server to fully start
    print("⏳ Waiting for server to start...")
    time.sleep(2)
    
    success = test_application()
    if not success:
        print("\n❌ Some tests failed. Please check the server logs.")
        exit(1) 