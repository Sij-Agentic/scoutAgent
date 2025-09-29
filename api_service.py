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
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        gcs_output_path=job.get("gcs_output_path"),
        error_message=job.get("error_message")
    )

async def process_job(job_id: str, job_request: JobRequest):
    """Process the ScoutAgent job in background"""
    try:
        # Update status to running
        jobs_db[job_id]["status"] = "running"
        
        # Get worker service URL (Cloud Run internal)
        worker_url = os.getenv("WORKER_SERVICE_URL", "http://worker-service:8080")
        
        # Call worker service
        async with httpx.AsyncClient(timeout=900.0) as client:  # 15 min timeout
            response = await client.post(f"{worker_url}/process", json={
                "job_id": job_id,
                "target_market": job_request.target_market,
                "keywords": job_request.keywords,
                "subreddits": job_request.subreddits,
                "per_query_limit": job_request.per_query_limit
            })
            
            if response.status_code == 200:
                result = response.json()
                jobs_db[job_id]["status"] = "completed"
                jobs_db[job_id]["gcs_output_path"] = result["gcs_output_path"]
            else:
                jobs_db[job_id]["status"] = "failed"
                jobs_db[job_id]["error_message"] = f"Worker service error: {response.text}"
                
    except Exception as e:
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error_message"] = str(e)
    
    finally:
        jobs_db[job_id]["completed_at"] = datetime.utcnow().isoformat()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
