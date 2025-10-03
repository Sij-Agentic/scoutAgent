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
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import uuid
import socket
import time
import requests
import threading

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
            # Set up ScoutAgent environment - stay in /app for consistent paths
            # os.chdir(temp_dir)  # Don't change directory - causes manifest path issues
            os.environ["PYTHONPATH"] = "/app"
            os.environ["SCOUT_TEMP_DIR"] = temp_dir  # Pass temp dir as env var if needed
            
            # Create progress log file in GCS
            gcs_prefix = f"scout/jobs/{request.job_id}/"
            progress_log_path = f"{gcs_prefix}progress.log"
            progress_blob = bucket.blob(progress_log_path)
            
            # Initialize progress log
            initial_log = f"""ScoutAgent Job Progress Log
Job ID: {request.job_id}
Started: {datetime.now().isoformat()}
Target Market: {request.target_market}
Keywords: {request.keywords}
Subreddits: {request.subreddits}
Per Query Limit: {request.per_query_limit}

Output will be available at: gs://{bucket_name}/{gcs_prefix}

================================================================================
PROGRESS LOG
================================================================================

"""
            progress_blob.upload_from_string(initial_log)
            print(f"Progress log created at: gs://{bucket_name}/{progress_log_path}")
            
            # Buffered, thread-safe progress logger to avoid GCS rate limit (429) on object mutations
            log_buffer: List[str] = []
            last_flush_ts = 0.0
            flush_interval = float(os.getenv("PROGRESS_FLUSH_INTERVAL", "5.0"))  # seconds
            max_buffer_lines = int(os.getenv("PROGRESS_MAX_BUFFER", "50"))
            log_lock = threading.Lock()

            def _flush_progress(force: bool = False):
                nonlocal last_flush_ts, log_buffer
                now = time.time()
                with log_lock:
                    if not force and (now - last_flush_ts < flush_interval) and len(log_buffer) < max_buffer_lines:
                        return
                    if not log_buffer:
                        return
                    payload = "\n".join(log_buffer) + "\n"
                    # Clear buffer before network I/O to minimize lock hold time
                    log_buffer.clear()
                    last_flush_ts = now
                try:
                    try:
                        current_content = progress_blob.download_as_string().decode('utf-8')
                    except Exception:
                        current_content = initial_log

                    updated_content = current_content + payload

                    # Retry with exponential backoff on transient errors (e.g., 429/5xx)
                    backoff = 0.5
                    for attempt in range(5):
                        try:
                            progress_blob.upload_from_string(updated_content)
                            break
                        except Exception:
                            try:
                                time.sleep(backoff)
                                backoff = min(backoff * 2, 10.0)
                            except Exception:
                                pass
                    else:
                        print("Failed to log to GCS after retries", file=sys.stderr)
                except Exception as e:
                    print(f"Failed to log to GCS: {e}", file=sys.stderr)

            def log_progress(message: str):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with log_lock:
                    log_buffer.append(f"[{timestamp}] {message}")
                _flush_progress(force=False)
            
            # Masked env var logging for diagnostics
            def _mask_prefix(value: str, prefix_len: int = 3) -> str:
                if not value:
                    return "<unset>"
                safe_prefix = value[:prefix_len]
                return f"{safe_prefix}***"

            print("Env check:")
            print(f"  SCOUT_OPENAI_API_KEY: {_mask_prefix(os.getenv('SCOUT_OPENAI_API_KEY'))}")
            print(f"  SCOUT_ANTHROPIC_API_KEY: {_mask_prefix(os.getenv('SCOUT_ANTHROPIC_API_KEY'))}")
            print(f"  SCOUT_GEMINI_API_KEY: {_mask_prefix(os.getenv('SCOUT_GEMINI_API_KEY'))}")
            print(f"  SCOUT_DEEPSEEK_API_KEY: {_mask_prefix(os.getenv('SCOUT_DEEPSEEK_API_KEY'))}")
            print(f"  SCOUT_REDDIT_CLIENT_ID: {_mask_prefix(os.getenv('SCOUT_REDDIT_CLIENT_ID'))}")
            print(f"  SCOUT_REDDIT_CLIENT_SECRET: {_mask_prefix(os.getenv('SCOUT_REDDIT_CLIENT_SECRET'))}")
            print(f"  SCOUT_REDDIT_USER_AGENT: {_mask_prefix(os.getenv('SCOUT_REDDIT_USER_AGENT'))}")

            # Preflight: verify outbound network and Reddit credentials by fetching an OAuth token
            try:
                reddit_client_id = os.getenv('SCOUT_REDDIT_CLIENT_ID')
                reddit_client_secret = os.getenv('SCOUT_REDDIT_CLIENT_SECRET')
                reddit_user_agent = os.getenv('SCOUT_REDDIT_USER_AGENT') or 'scout-agent/1.0'

                if reddit_client_id and reddit_client_secret:
                    print("Reddit preflight: requesting OAuth token...")
                    auth = requests.auth.HTTPBasicAuth(reddit_client_id, reddit_client_secret)
                    headers = {"User-Agent": reddit_user_agent}
                    data = {"grant_type": "client_credentials"}
                    token_resp = requests.post(
                        "https://www.reddit.com/api/v1/access_token",
                        auth=auth,
                        headers=headers,
                        data=data,
                        timeout=15,
                    )
                    print(f"Reddit preflight token status: {token_resp.status_code}")
                    if token_resp.ok:
                        token_json = token_resp.json()
                        access_token = token_json.get("access_token")
                        token_type = token_json.get("token_type", "bearer").capitalize()
                        if access_token:
                            print("Reddit preflight: token acquired, testing API GET...")
                            api_headers = {
                                "Authorization": f"{token_type} {access_token}",
                                "User-Agent": reddit_user_agent,
                            }
                            api_resp = requests.get(
                                "https://oauth.reddit.com/r/popular?limit=1",
                                headers=api_headers,
                                timeout=15,
                            )
                            print(f"Reddit preflight API status: {api_resp.status_code}")
                            if not api_resp.ok:
                                print(f"Reddit preflight API body: {api_resp.text[:300]}")
                        else:
                            print("Reddit preflight: token missing in response body")
                    else:
                        print(f"Reddit preflight token body: {token_resp.text[:300]}")
                else:
                    print("Reddit preflight: client id/secret not set; skipping")
            except requests.Timeout:
                print("Reddit preflight: network timeout (check egress/VPC)")
            except Exception as preflight_exc:
                print(f"Reddit preflight: unexpected error: {preflight_exc}")

            # Helper: wait for TCP port
            def wait_for_port(host: str, port: int, timeout_seconds: int = 120) -> bool:
                deadline = time.time() + timeout_seconds
                while time.time() < deadline:
                    try:
                        with socket.create_connection((host, port), timeout=2):
                            return True
                    except OSError:
                        time.sleep(1)
                return False

            # Log start
            log_progress("Starting MCP servers...")
            
            # Start MCP servers in background
            mcp_processes = []
            try:
                mcp_defs = [
                    ("gap_finder_tools", ["python", "-m", "scout_agent.mcp_integration.server.gap_finder_tools"], 8000),
                    ("reddit_api", ["python", "-m", "scout_agent.mcp_integration.server.reddit_api"], 8001),
                    ("research_tools", ["python", "-m", "scout_agent.mcp_integration.server.research_tools"], 8002),
                    ("web_search", ["python", "-m", "scout_agent.mcp_integration.server.web_search"], 8004),
                ]

                # Stream MCP server logs (controlled by env var)
                verbose_mcp_logs = os.getenv("VERBOSE_MCP_LOGS", "false").lower() == "true"
                
                def _stream_mcp(pipe, writer, prefix: str, is_stderr: bool = False):
                    for line in iter(pipe.readline, ""):
                        writer.write(line)
                        writer.flush()
                        # Only print to console if verbose mode enabled
                        if verbose_mcp_logs:
                            try:
                                if is_stderr:
                                    print(f"[{prefix}][STDERR] {line}", file=sys.stderr, end="")
                                else:
                                    print(f"[{prefix}][STDOUT] {line}", end="")
                            except Exception:
                                pass
                    pipe.close()

                for name, cmd, port in mcp_defs:
                    print(f"Starting MCP server '{name}' on port {port} with: {' '.join(cmd)}")
                    # Prepare per-server log files
                    mcp_log_dir = os.path.join(temp_dir, "worker_logs", f"mcp_{name}")
                    os.makedirs(mcp_log_dir, exist_ok=True)
                    stdout_file = open(os.path.join(mcp_log_dir, "stdout.log"), "w")
                    stderr_file = open(os.path.join(mcp_log_dir, "stderr.log"), "w")

                    p = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                        env={**os.environ, "PYTHONUNBUFFERED": "1"}
                    )

                    # Start log streaming threads
                    import threading
                    t_out = threading.Thread(target=_stream_mcp, args=(p.stdout, stdout_file, name, False), daemon=True)
                    t_err = threading.Thread(target=_stream_mcp, args=(p.stderr, stderr_file, name, True), daemon=True)
                    t_out.start()
                    t_err.start()

                    mcp_processes.append((name, port, p, stdout_file, stderr_file))

                # Wait for all ports
                for name, port, *_ in mcp_processes:
                    print(f"Waiting for MCP server '{name}' to be ready on port {port}...")
                    log_progress(f"Starting MCP server: {name} on port {port}")
                    if not wait_for_port("127.0.0.1", port, timeout_seconds=120):
                        log_progress(f"ERROR: MCP server '{name}' failed to start")
                        raise Exception(f"MCP server '{name}' failed to start on port {port}")

                    # Basic HTTP health check on root
                    try:
                        resp = requests.get(f"http://127.0.0.1:{port}/", timeout=5)
                        print(f"MCP '{name}' GET / -> {resp.status_code}")
                    except Exception as http_exc:
                        print(f"MCP '{name}' GET / failed: {http_exc}")

                    # Skip SSE health check - SSE endpoints are designed to stream indefinitely
                    print(f"MCP '{name}' health check passed (port {port} is responding)")

                print(f"All MCP servers started successfully")
                log_progress("All MCP servers ready")
                log_progress(f"Starting ScoutAgent workflow...")
                
                # Run ScoutAgent main.py to let servers fully initialize routes/workers
                time.sleep(10)
            except Exception as e:
                # If any MCP startup fails, ensure we terminate what we started
                for item in mcp_processes:
                    p = item[2]
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
                    try:
                        for line in iter(pipe.readline, ""):
                            try:
                                writer.write(line)
                                writer.flush()
                            except (ValueError, OSError):
                                # File closed, stop streaming
                                break
                            try:
                                if is_stderr:
                                    print(line, file=sys.stderr, end="")
                                else:
                                    print(line, end="")
                                
                                # Log important lines to GCS
                                line_lower = line.lower()
                                # Reduce volume: drop generic 'info' to avoid excessive writes
                                if any(keyword in line_lower for keyword in ['starting', 'completed', 'error', 'warning', 'stage']):
                                    log_progress(line.strip())
                            except Exception:
                                pass
                    finally:
                        try:
                            pipe.close()
                        except Exception:
                            pass

                # Read stdout and stderr concurrently
                import threading
                t_out = threading.Thread(target=_stream, args=(process.stdout, f_out, False))
                t_err = threading.Thread(target=_stream, args=(process.stderr, f_err, True))
                t_out.start()
                t_err.start()
                # Wait for process with extended timeout (1 hour for long jobs)
                return_code = process.wait(timeout=3600)
                t_out.join()
                t_err.join()
            
            if return_code != 0:
                log_progress(f"ERROR: ScoutAgent failed with exit code {return_code}")
                # Read tail of logs for error message context
                tail_err = ""
                tail_out = ""
                try:
                    with open(stderr_path, "r") as f:
                        lines = f.readlines()
                        tail_err = "".join(lines[-50:])
                except Exception:
                    pass
                try:
                    with open(stdout_path, "r") as f:
                        lines = f.readlines()
                        tail_out = "".join(lines[-50:])
                except Exception:
                    pass
                
                error_msg = f"ScoutAgent failed with exit code {return_code}\n"
                error_msg += f"\n=== STDERR (last 50 lines) ===\n{tail_err}\n"
                error_msg += f"\n=== STDOUT (last 50 lines) ===\n{tail_out}\n"
                print(error_msg, file=sys.stderr)
                # Ensure any pending progress lines are flushed before raising
                try:
                    _flush_progress(force=True)
                except Exception:
                    pass
                raise Exception(f"ScoutAgent failed (code {return_code})")

            log_progress("ScoutAgent workflow completed successfully")
            log_progress("Uploading results to GCS...")
            
            # Clean up MCP servers
            for item in mcp_processes:
                name, port, p = item[:3]
                try:
                    print(f"Stopping MCP server '{name}' (port {port})")
                    p.terminate()
                except Exception:
                    pass
                # Close log files
                try:
                    if len(item) >= 5:
                        item[3].close()
                        item[4].close()
                except Exception:
                    pass
            
            # Find the output directory (usually /app/data/runs/)
            output_dir = None
            # Check /app/data/runs first (where manifest is created)
            app_data_runs = "/app/data/runs"
            if os.path.exists(app_data_runs) and os.path.isdir(app_data_runs):
                output_dir = app_data_runs
                print(f"Found output directory at: {output_dir}")
            else:
                # Fallback: search current directory
                for root, dirs, files in os.walk("."):
                    if "data" in root and "runs" in root:
                        output_dir = root
                        print(f"Found output directory at: {output_dir}")
                        break
            
            if not output_dir:
                print(f"ERROR: No output directory found. Checked: {app_data_runs} and current dir")
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
            
            print(f"Job {request.job_id} completed successfully")
            print(f"Output uploaded to: {gcs_output_path}")
            print(f"Total files uploaded: {len(uploaded_files)}")
            
            log_progress(f"Upload complete: {len(uploaded_files)} files")
            log_progress(f"Results available at: {gcs_output_path}")
            log_progress("Job completed successfully!")
            # Force-flush any buffered logs before writing status
            _flush_progress(force=True)
            
            # Write status file to GCS for API to detect completion
            import json
            status_data = {
                "job_id": request.job_id,
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "gcs_output_path": gcs_output_path,
                "files_uploaded": len(uploaded_files),
                "error": None
            }
            status_blob = bucket.blob(f"{gcs_prefix}job_status.json")
            status_blob.upload_from_string(json.dumps(status_data, indent=2), content_type="application/json")
            print(f"Status file written to: gs://{bucket_name}/{gcs_prefix}job_status.json")
            
            return ProcessResponse(
                job_id=request.job_id,
                status="completed",
                gcs_output_path=gcs_output_path
            )
            
    except subprocess.TimeoutExpired as e:
        error_msg = f"Job {request.job_id} timed out after {e.timeout}s"
        print(error_msg, file=sys.stderr)
        
        # Flush any buffered progress before writing failure status
        try:
            _flush_progress(force=True)
        except Exception:
            pass
        # Write failure status to GCS
        try:
            # Ensure bucket is available even if initialization failed earlier
            if 'bucket' not in locals() or bucket is None:
                try:
                    bucket_name = os.getenv("GCS_BUCKET", "scout-agent-outputs")
                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                except Exception:
                    bucket = None

            import json
            status_data = {
                "job_id": request.job_id,
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error": error_msg,
                "error_type": "timeout"
            }
            gcs_prefix = f"scout/jobs/{request.job_id}/"
            if bucket is not None:
                status_blob = bucket.blob(f"{gcs_prefix}job_status.json")
                status_blob.upload_from_string(json.dumps(status_data, indent=2), content_type="application/json")
        except Exception:
            pass
        
        raise HTTPException(status_code=408, detail=error_msg)
    except Exception as e:
        error_msg = f"Error processing job {request.job_id}: {str(e)}"
        print(error_msg, file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        # Write failure status to GCS
        try:
            # Ensure bucket is available even if initialization failed earlier
            if 'bucket' not in locals() or bucket is None:
                try:
                    bucket_name = os.getenv("GCS_BUCKET", "scout-agent-outputs")
                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                except Exception:
                    bucket = None

            import json
            status_data = {
                "job_id": request.job_id,
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error": str(e),
                "error_type": "exception"
            }
            gcs_prefix = f"scout/jobs/{request.job_id}/"
            if bucket is not None:
                status_blob = bucket.blob(f"{gcs_prefix}job_status.json")
                status_blob.upload_from_string(json.dumps(status_data, indent=2), content_type="application/json")
        except Exception:
            pass
        
        raise HTTPException(status_code=500, detail=f"Job processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
