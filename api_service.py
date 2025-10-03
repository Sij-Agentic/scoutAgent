#!/usr/bin/env python3
"""
ScoutAgent API Service for Cloud Run
Handles job creation and status checking
"""

import uuid
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import os
import httpx
import io
import zipfile

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
    progress_url: str
    output_location: str

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
    
    # Generate progress URL (public endpoint)
    api_base = os.getenv("API_BASE_URL", "https://scout-agent-511946707043.us-central1.run.app")
    progress_url = f"{api_base}/jobs/{job_id}/progress"
    
    # Output location (predictable)
    bucket_name = os.getenv("GCS_BUCKET", "scout-agent-outputs")
    output_location = f"gs://{bucket_name}/scout/jobs/{job_id}/"
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=jobs_db[job_id]["created_at"],
        progress_url=progress_url,
        output_location=output_location
    )

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status and results"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    
    # If job is still running, check GCS for status file (worker writes this on completion/failure)
    if job["status"] == "running":
        try:
            from google.cloud import storage
            import json
            bucket_name = os.getenv("GCS_BUCKET", "scout-agent-outputs")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            # Check for status file written by worker
            status_blob = bucket.blob(f"scout/jobs/{job_id}/job_status.json")
            if status_blob.exists():
                # Read status file
                status_data = json.loads(status_blob.download_as_string())
                job["status"] = status_data.get("status", "completed")
                job["completed_at"] = status_data.get("completed_at", datetime.utcnow().isoformat())
                job["gcs_output_path"] = status_data.get("gcs_output_path", f"gs://{bucket_name}/scout/jobs/{job_id}/")
                if status_data.get("error"):
                    job["error_message"] = status_data.get("error")
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

@app.get("/jobs/{job_id}/progress")
async def get_job_progress(job_id: str):
    """Get real-time progress log for a job - publicly accessible"""
    # Note: This endpoint is public (no auth check) so users can watch progress
    # Job IDs are UUIDs (hard to guess), and logs don't contain sensitive data
    
    try:
        from google.cloud import storage
        bucket_name = os.getenv("GCS_BUCKET", "scout-agent-outputs")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Read progress log
        progress_blob = bucket.blob(f"scout/jobs/{job_id}/progress.log")
        if not progress_blob.exists():
            return {
                "job_id": job_id,
                "progress": "Progress log not yet available. Job may be starting...",
                "status": "pending"
            }
        
        progress_content = progress_blob.download_as_string().decode('utf-8')
        
        # Check if job completed
        status = "running"
        if "Job completed successfully!" in progress_content:
            status = "completed"
        elif "ERROR:" in progress_content:
            status = "failed"
        
        return {
            "job_id": job_id,
            "progress": progress_content,
            "status": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch progress: {str(e)}")

@app.get("/jobs/{job_id}/download")
async def download_job_results(job_id: str):
    """Download job results as a zip file"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    
    # Check if job is completed
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed yet. Current status: {job['status']}")
    
    if not job.get("gcs_output_path"):
        raise HTTPException(status_code=404, detail="No output path found for this job")
    
    try:
        from google.cloud import storage
        bucket_name = os.getenv("GCS_BUCKET", "scout-agent-outputs")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Create zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # List all files in the job directory
            prefix = f"scout/jobs/{job_id}/"
            blobs = bucket.list_blobs(prefix=prefix)
            
            for blob in blobs:
                # Get relative path
                relative_path = blob.name[len(prefix):]
                if relative_path:  # Skip the directory itself
                    # Download blob content
                    content = blob.download_as_bytes()
                    # Add to zip
                    zip_file.writestr(relative_path, content)
        
        # Seek to beginning of buffer
        zip_buffer.seek(0)
        
        # Return as streaming response
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=scout_job_{job_id}.zip"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download results: {str(e)}")

async def process_job(job_id: str, job_request: JobRequest):
    """Process the ScoutAgent job in background - fire and forget"""
    try:
        # Update status to running
        jobs_db[job_id]["status"] = "running"
        
        # Get worker service URL (Cloud Run internal)
        worker_url = os.getenv("WORKER_SERVICE_URL", "http://worker-service:8080")
        
        # Fire-and-forget: Don't wait for response, let worker run independently
        # Worker will write status file to GCS when complete
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
