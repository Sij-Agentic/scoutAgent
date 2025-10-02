#!/usr/bin/env python3
"""
ScoutAgent API Service for Cloud Run
Handles job creation and status checking
"""

import os
import uuid
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import httpx

app = FastAPI(title="ScoutAgent API", version="1.0.0")

# Job storage (in production, use Firestore or similar)
jobs_db: Dict[str, Dict[str, Any]] = {}

class JobRequest(BaseModel):
    target_market: str
    keywords: str
    subreddits: str
    per_query_limit: int = 2

class JobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    estimated_duration: str = "5-15 minutes"

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    created_at: str
    completed_at: Optional[str] = None
    gcs_output_path: Optional[str] = None
    error_message: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "ScoutAgent API Service", "version": "1.0.0"}

@app.post("/jobs", response_model=JobResponse)
async def create_job(job_request: JobRequest, background_tasks: BackgroundTasks):
    """Create a new ScoutAgent job"""
    job_id = str(uuid.uuid4())
    
    # Store job info
    jobs_db[job_id] = {
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "target_market": job_request.target_market,
        "keywords": job_request.keywords,
        "subreddits": job_request.subreddits,
        "per_query_limit": job_request.per_query_limit,
        "gcs_output_path": None,
        "error_message": None
    }
    
    # Trigger background job processing
    background_tasks.add_task(process_job, job_id, job_request)
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=jobs_db[job_id]["created_at"]
    )

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status and results"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    
    # If job is still running, check GCS to see if output exists (worker completed)
    if job["status"] == "running":
        try:
            from google.cloud import storage
            bucket_name = os.getenv("GCS_BUCKET", "scout-agent-outputs")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            # Check if manifest file exists in GCS
            manifest_blob = bucket.blob(f"scout/jobs/{job_id}/data/runs/{job_id}/run_manifest.json")
            if manifest_blob.exists():
                # Job completed, update status
                job["status"] = "completed"
                job["gcs_output_path"] = f"gs://{bucket_name}/scout/jobs/{job_id}/"
                job["completed_at"] = datetime.utcnow().isoformat()
        except Exception as e:
            # Ignore GCS check errors, return current status
            pass
    
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        gcs_output_path=job.get("gcs_output_path"),
        error_message=job.get("error_message")
    )

async def process_job(job_id: str, job_request: JobRequest):
    """Process the ScoutAgent job in background - fire and forget"""
    try:
        # Update status to running
        jobs_db[job_id]["status"] = "running"
        
        # Get worker service URL (Cloud Run internal)
        worker_url = os.getenv("WORKER_SERVICE_URL", "http://worker-service:8080")
        
        # Fire-and-forget: Don't wait for response, let worker run independently
        # Worker will need to update status via callback or we poll GCS
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, read=None)) as client:
            try:
                # Send request but don't wait for completion
                await client.post(f"{worker_url}/process", json={
                    "job_id": job_id,
                    "target_market": job_request.target_market,
                    "keywords": job_request.keywords,
                    "subreddits": job_request.subreddits,
                    "per_query_limit": job_request.per_query_limit
                }, timeout=10.0)  # Short timeout just to confirm worker received it
            except httpx.TimeoutException:
                # Expected - worker is processing, we don't wait
                pass
            except httpx.ReadTimeout:
                # Expected - worker is processing, we don't wait
                pass
                
    except Exception as e:
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error_message"] = f"Failed to start job: {str(e)}"
        jobs_db[job_id]["completed_at"] = datetime.utcnow().isoformat()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
