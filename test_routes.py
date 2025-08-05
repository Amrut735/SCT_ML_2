#!/usr/bin/env python3
"""
Test script to verify Flask routes are working
"""

import requests
import json

def test_routes():
    """Test the main application routes"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Flask Routes...")
    print("=" * 40)
    
    # Test 1: Check if server is running
    print("1. Testing server connectivity...")
    try:
        response = requests.get(base_url, timeout=5)
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Length: {len(response.text)}")
        if response.status_code == 200:
            print("   ✅ Server is responding!")
        else:
            print(f"   ❌ Server returned status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Cannot connect to server: {e}")
        return False
    
    # Test 2: Test sample data endpoint
    print("\n2. Testing sample data endpoint...")
    try:
        response = requests.post(f"{base_url}/use_sample_data", timeout=10)
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Length: {len(response.text)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("   ✅ JSON response received!")
                print(f"   Success: {data.get('success', False)}")
                if data.get('success'):
                    print(f"   Total customers: {data.get('data_summary', {}).get('total_customers', 'N/A')}")
                else:
                    print(f"   Error: {data.get('error', 'Unknown error')}")
            except json.JSONDecodeError as e:
                print(f"   ❌ Invalid JSON response: {e}")
                print(f"   Response preview: {response.text[:200]}...")
        else:
            print(f"   ❌ Request failed with status: {response.status_code}")
            print(f"   Response preview: {response.text[:200]}...")
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("🎉 Route testing completed!")
    return True

if __name__ == "__main__":
    test_routes() 