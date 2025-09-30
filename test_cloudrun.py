#!/usr/bin/env python3
"""
Test script for ScoutAgent Cloud Run API
"""

import requests
import time
import json

# Configuration
API_URL = "https://scout-agent-511946707043.us-central1.run.app"
BUCKET_NAME = "scout-agent-outputs"

def test_job_creation():
    """Test creating a job"""
    print("🧪 Testing job creation...")
    
    job_data = {
        "target_market": "Knowledge management tools",
        "keywords": "bidirectional links,markdown sync",
        "subreddits": "PKMS,productivity,Evernote",
        "per_query_limit": 2
    }
    
    response = requests.post(f"{API_URL}/jobs", json=job_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Job created: {result['job_id']}")
        return result['job_id']
    else:
        print(f"❌ Job creation failed: {response.status_code} - {response.text}")
        return None

def test_job_status(job_id):
    """Test checking job status"""
    print(f"🔍 Checking status for job: {job_id}")
    
    response = requests.get(f"{API_URL}/jobs/{job_id}")
    
    if response.status_code == 200:
        status = response.json()
        print(f"📊 Status: {status['status']}")
        if status.get('gcs_output_path'):
            print(f"📁 Output: {status['gcs_output_path']}")
        if status.get('error_message'):
            print(f"❌ Error: {status['error_message']}")
        return status
    else:
        print(f"❌ Status check failed: {response.status_code} - {response.text}")
        return None

def main():
    print("🚀 ScoutAgent Cloud Run API Test")
    print(f"API URL: {API_URL}")
    print()
    
    # Test job creation
    job_id = test_job_creation()
    if not job_id:
        return
    
    print()
    print("⏳ Waiting for job to complete...")
    
    # Poll for completion
    max_attempts = 30  # 5 minutes
    for attempt in range(max_attempts):
        status = test_job_status(job_id)
        if status and status['status'] in ['completed', 'failed']:
            break
        time.sleep(10)
    
    print()
    if status:
        if status['status'] == 'completed':
            print("🎉 Job completed successfully!")
            print(f"📁 Results: {status.get('gcs_output_path', 'Not available')}")
        else:
            print("❌ Job failed")
            print(f"Error: {status.get('error_message', 'Unknown error')}")
    else:
        print("⏰ Job timed out")

if __name__ == "__main__":
    main()
