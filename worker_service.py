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
import socket
import time

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
            
            # Helper: wait for TCP port
            def wait_for_port(host: str, port: int, timeout_seconds: int = 60) -> bool:
                deadline = time.time() + timeout_seconds
                while time.time() < deadline:
                    try:
                        with socket.create_connection((host, port), timeout=2):
                            return True
                    except OSError:
                        time.sleep(1)
                return False

            # Start MCP servers in background
            mcp_processes = []
            try:
                mcp_defs = [
                    ("gap_finder_tools", ["python", "-m", "scout_agent.mcp_integration.server.gap_finder_tools"], 8000),
                    ("reddit_api", ["python", "-m", "scout_agent.mcp_integration.server.reddit_api"], 8001),
                    ("research_tools", ["python", "-m", "scout_agent.mcp_integration.server.research_tools"], 8002),
                    ("web_search", ["python", "-m", "scout_agent.mcp_integration.server.web_search"], 8004),
                ]

                for name, cmd, port in mcp_defs:
                    print(f"Starting MCP server '{name}' on port {port} with: {' '.join(cmd)}")
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
                    mcp_processes.append((name, port, p))

                # Wait for all ports
                for name, port, _ in mcp_processes:
                    print(f"Waiting for MCP server '{name}' to be ready on port {port}...")
                    if not wait_for_port("127.0.0.1", port, timeout_seconds=90):
                        raise Exception(f"MCP server '{name}' failed to start on port {port}")
                print("All MCP servers are up.")
            except Exception:
                # If any MCP startup fails, ensure we terminate what we started
                for _, _, p in mcp_processes:
                    try:
                        p.terminate()
                    except Exception:
                        pass
                raise

            # Run ScoutAgent workflow
            cmd = [
                "python", "-m", "scout_agent.main",
                "--target-market", request.target_market,
                "--keywords", request.keywords,
                "--subreddits", request.subreddits,
                "--per-query-limit", str(request.per_query_limit)
            ]
            
            print(f"Running ScoutAgent command: {' '.join(cmd)}")

            # Stream stdout/stderr live to Cloud Run logs and to files
            log_dir = os.path.join(temp_dir, "worker_logs")
            os.makedirs(log_dir, exist_ok=True)
            stdout_path = os.path.join(log_dir, "stdout.log")
            stderr_path = os.path.join(log_dir, "stderr.log")

            with open(stdout_path, "w") as f_out, open(stderr_path, "w") as f_err:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"}
                )

                def _stream(pipe, writer, is_stderr: bool = False):
                    for line in iter(pipe.readline, ""):
                        writer.write(line)
                        writer.flush()
                        try:
                            if is_stderr:
                                print(line, file=sys.stderr, end="")
                            else:
                                print(line, end="")
                        except Exception:
                            pass
                    pipe.close()

                # Read stdout and stderr concurrently
                import threading
                t_out = threading.Thread(target=_stream, args=(process.stdout, f_out, False))
                t_err = threading.Thread(target=_stream, args=(process.stderr, f_err, True))
                t_out.start()
                t_err.start()
                return_code = process.wait(timeout=900)
                t_out.join()
                t_err.join()
            
            if return_code != 0:
                # Read tail of logs for error message context
                tail_err = ""
                try:
                    with open(stderr_path, "r") as f:
                        lines = f.readlines()
                        tail_err = "".join(lines[-50:])
                except Exception:
                    pass
                raise Exception(f"ScoutAgent failed (code {return_code}): {tail_err}")

            # Clean up MCP servers
            for name, port, p in mcp_processes:
                try:
                    print(f"Stopping MCP server '{name}' (port {port})")
                    p.terminate()
                except Exception:
                    pass
            
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
            
            # Also upload debug and logs (and worker_logs) if they exist
            for folder in ["debug", "logs", "worker_logs"]:
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
