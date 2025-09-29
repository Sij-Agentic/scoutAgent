#!/usr/bin/env python3
"""
ScoutAgent Worker Service for Cloud Run
Processes ScoutAgent jobs and uploads results to GCS
"""

import os
import sys
import subprocess
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import uuid

app = FastAPI(title="ScoutAgent Worker", version="1.0.0")

class ProcessRequest(BaseModel):
    job_id: str
    target_market: str
    keywords: str
    subreddits: str
    per_query_limit: int

class ProcessResponse(BaseModel):
    job_id: str
    status: str
    gcs_output_path: str

@app.get("/")
async def root():
    return {"message": "ScoutAgent Worker Service", "version": "1.0.0"}

@app.post("/process", response_model=ProcessResponse)
async def process_job(request: ProcessRequest):
    """Process a ScoutAgent job and upload results to GCS"""
    try:
        # Get GCS bucket name
        bucket_name = os.getenv("GCS_BUCKET")
        if not bucket_name:
            raise HTTPException(status_code=500, detail="GCS_BUCKET environment variable not set")
        
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Create temporary directory for this job
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up ScoutAgent environment
            os.chdir(temp_dir)
            os.environ["PYTHONPATH"] = "/app"
            
            # Run ScoutAgent workflow
            cmd = [
                "python", "-m", "scout_agent.main",
                "--target-market", request.target_market,
                "--keywords", request.keywords,
                "--subreddits", request.subreddits,
                "--per-query-limit", str(request.per_query_limit)
            ]
            
            print(f"Running ScoutAgent command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            
            if result.returncode != 0:
                raise Exception(f"ScoutAgent failed: {result.stderr}")
            
            # Find the output directory (usually data/runs/)
            output_dir = None
            for root, dirs, files in os.walk("."):
                if "data" in root and "runs" in root:
                    output_dir = root
                    break
            
            if not output_dir:
                raise Exception("No output directory found")
            
            # Upload to GCS
            gcs_prefix = f"scout/jobs/{request.job_id}/"
            uploaded_files = []
            
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, output_dir)
                    gcs_path = f"{gcs_prefix}{relative_path}"
                    
                    blob = bucket.blob(gcs_path)
                    blob.upload_from_filename(local_path)
                    uploaded_files.append(gcs_path)
            
            # Also upload debug and logs if they exist
            for folder in ["debug", "logs"]:
                if os.path.exists(folder):
                    for root, dirs, files in os.walk(folder):
                        for file in files:
                            local_path = os.path.join(root, file)
                            relative_path = os.path.relpath(local_path, ".")
                            gcs_path = f"{gcs_prefix}{relative_path}"
                            
                            blob = bucket.blob(gcs_path)
                            blob.upload_from_filename(local_path)
                            uploaded_files.append(gcs_path)
            
            gcs_output_path = f"gs://{bucket_name}/{gcs_prefix}"
            
            return ProcessResponse(
                job_id=request.job_id,
                status="completed",
                gcs_output_path=gcs_output_path
            )
            
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Job processing timed out")
    except Exception as e:
        print(f"Error processing job {request.job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Job processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
