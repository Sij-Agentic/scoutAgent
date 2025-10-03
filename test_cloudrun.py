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
        print(f"\n✅ Job created successfully!")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📋 Job ID: {result['job_id']}")
        print(f"⏱️  Estimated Duration: {result.get('estimated_duration', '5-15 minutes')}")
        print(f"")
        print(f"📊 Watch Progress (Public - Shareable!):")
        print(f"   {result.get('progress_url', 'N/A')}")
        print(f"")
        print(f"📁 Final Output Location:")
        print(f"   {result.get('output_location', 'N/A')}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"\n💡 Tip: Share the progress URL with your team to watch together!")
        print(f"💡 Progress log available for 24 hours")
        return result
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
    result = test_job_creation()
    if not result:
        return
    
    job_id = result['job_id']
    progress_url = result.get('progress_url')
    output_location = result.get('output_location')
    
    print()
    print("⏳ Waiting for job to complete...")
    if progress_url:
        print(f"💡 Watch live: watch -n 5 \"curl -s {progress_url} | jq -r '.progress'\"")
    
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
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("🎉 Job completed successfully!")
            print(f"")
            print(f"📁 Download Results:")
            print(f"   curl \"{API_URL}/jobs/{job_id}/download\" -o results.zip")
            print(f"")
            print(f"📂 Or access directly from GCS:")
            print(f"   gsutil -m cp -r \"{status.get('gcs_output_path', output_location)}\" ./results/")
            print(f"")
            print(f"📊 View Progress Log:")
            print(f"   {progress_url}")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        else:
            print("❌ Job failed")
            print(f"Error: {status.get('error_message', 'Unknown error')}")
            if progress_url:
                print(f"\n📊 Check progress log for details:")
                print(f"   {progress_url}")
    else:
        print("⏰ Job timed out")
        if progress_url:
            print(f"\n📊 Check progress log:")
            print(f"   {progress_url}")

if __name__ == "__main__":
    main()
