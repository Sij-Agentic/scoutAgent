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
    """Deprecated: status polling removed to avoid confusion."""
    pass

def main():
    print("🚀 ScoutAgent Cloud Run API Test")
    print(f"API URL: {API_URL}")
    print()
    
    # Create a job
    result = test_job_creation()
    if not result:
        return
    
    job_id = result['job_id']
    progress_url = result.get('progress_url')
    output_location = result.get('output_location')
    
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 Watch Progress (Public - Shareable):")
    print(f"   {progress_url}")
    print("🟢 Live tail:")
    print(f"   watch -n 5 \"curl -s {progress_url} | jq -r '.progress'\"")
    print("")
    print("📁 Final Output Location:")
    print(f"   {output_location}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Do not poll status; users can watch progress and fetch outputs when ready.

if __name__ == "__main__":
    main()
