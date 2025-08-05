#!/usr/bin/env python3
"""
API Testing Script
Test the Prompt Injection Detection API endpoints
"""

import requests
import json
import os
import time
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"
TEST_FILES_DIR = Path(__file__).parent.parent / "data" / "attack_samples"

def test_health_endpoint():
    """Test health check endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{API_BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check passed")
        print(f"   Status: {data['status']}")
        print(f"   Detectors loaded: {data['detectors_loaded']}")
        print(f"   Uptime: {data['uptime']:.2f}s")
    else:
        print(f"❌ Health check failed: {response.status_code}")
    print()

def test_metrics_endpoint():
    """Test metrics endpoint"""
    print("📊 Testing metrics endpoint...")
    response = requests.get(f"{API_BASE_URL}/metrics")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Metrics endpoint working")
        print(f"   Total requests: {data['total_requests']}")
        print(f"   Successful detections: {data['successful_detections']}")
        print(f"   Average processing time: {data['average_processing_time']:.3f}s")
    else:
        print(f"❌ Metrics failed: {response.status_code}")
    print()

def test_single_file_detection():
    """Test single file detection"""
    print("📄 Testing single file detection...")
    
    # Find a test file
    test_files = list(TEST_FILES_DIR.glob("*.pdf"))
    if not test_files:
        print("❌ No test files found in data/attack_samples/")
        return
    
    test_file = test_files[0]
    print(f"   Using file: {test_file.name}")
    
    start_time = time.time()
    
    with open(test_file, 'rb') as f:
        files = {'file': (test_file.name, f, 'application/pdf')}
        params = {
            'detector_type': 'ensemble',
            'return_details': True
        }
        
        response = requests.post(
            f"{API_BASE_URL}/detect/single",
            files=files,
            params=params
        )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Single file detection successful")
        print(f"   File: {data['file_name']}")
        print(f"   Malicious: {data['is_malicious']}")
        print(f"   Risk Score: {data['risk_score']:.3f}")
        print(f"   Detection Count: {data['detection_count']}")
        print(f"   Detection Types: {data['detection_types']}")
        print(f"   API Processing Time: {data['processing_time']:.3f}s")
        print(f"   Total Request Time: {processing_time:.3f}s")
    else:
        print(f"❌ Single file detection failed: {response.status_code}")
        print(f"   Error: {response.text}")
    print()

def test_batch_detection():
    """Test batch file detection"""
    print("📚 Testing batch file detection...")
    
    # Find test files
    test_files = list(TEST_FILES_DIR.glob("*.pdf"))[:3]  # Use first 3 files
    if len(test_files) < 2:
        print("❌ Need at least 2 test files for batch testing")
        return
    
    print(f"   Using {len(test_files)} files")
    
    start_time = time.time()
    
    files = []
    for test_file in test_files:
        files.append(('files', (test_file.name, open(test_file, 'rb'), 'application/pdf')))
    
    params = {
        'detector_type': 'ensemble',
        'return_details': False,
        'max_files': 10
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/detect/batch",
            files=files,
            params=params
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch detection successful")
            print(f"   Total files: {data['total_files']}")
            print(f"   Malicious files: {data['malicious_files']}")
            print(f"   Clean files: {data['clean_files']}")
            print(f"   API Processing Time: {data['processing_time']:.3f}s")
            print(f"   Total Request Time: {processing_time:.3f}s")
            print(f"   Files per second: {data['summary']['files_per_second']:.2f}")
            print(f"   Malicious rate: {data['summary']['malicious_rate']:.2%}")
        else:
            print(f"❌ Batch detection failed: {response.status_code}")
            print(f"   Error: {response.text}")
    
    finally:
        # Close file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()
    
    print()

def test_error_handling():
    """Test error handling with invalid requests"""
    print("🚨 Testing error handling...")
    
    # Test with non-PDF file
    print("   Testing non-PDF file...")
    files = {'file': ('test.txt', b'This is not a PDF', 'text/plain')}
    response = requests.post(f"{API_BASE_URL}/detect/single", files=files)
    
    if response.status_code == 400:
        print("   ✅ Correctly rejected non-PDF file")
    else:
        print(f"   ❌ Expected 400, got {response.status_code}")
    
    # Test with invalid detector type
    print("   Testing invalid detector type...")
    test_files = list(TEST_FILES_DIR.glob("*.pdf"))
    if test_files:
        with open(test_files[0], 'rb') as f:
            files = {'file': (test_files[0].name, f, 'application/pdf')}
            params = {'detector_type': 'invalid'}
            response = requests.post(f"{API_BASE_URL}/detect/single", files=files, params=params)
        
        if response.status_code == 400:
            print("   ✅ Correctly rejected invalid detector type")
        else:
            print(f"   ❌ Expected 400, got {response.status_code}")
    
    print()

def main():
    """Run all tests"""
    print("🧪 Starting API Tests")
    print("=" * 50)
    
    try:
        # Test basic endpoints
        test_health_endpoint()
        test_metrics_endpoint()
        
        # Test detection endpoints
        test_single_file_detection()
        test_batch_detection()
        
        # Test error handling
        test_error_handling()
        
        print("🎉 All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    main()