## ScoutAgent on Google Cloud Run

A containerized, agentic research system (ScoutAgent) deployed on Google Cloud Run with Google Cloud Storage (GCS) for outputs. This guide covers building, deploying, running, testing, and retrieving results without modifying the core `scout_agent/main.py`.

### Overview
- **API service** receives job requests and returns a `job_id` for polling.
- **Worker service** runs the long job and uploads artifacts to GCS at `gs://<bucket>/scout/jobs/<job_id>/...`.
- Both services use the same container image and run different entrypoints.
- No edits to `scout_agent/main.py`.

### Repository Layout
- `scout_agent/` — original app (unchanged main entry `scout_agent/main.py`).
- `api_service.py` — FastAPI API to create jobs and check status.
- `worker_service.py` — FastAPI worker that runs the job and uploads to GCS.
- `Dockerfile` — multi-stage build for the container image.
- `requirements-docker.txt` — pinned dependencies for container builds.
- `start_services.sh`, `start_mcp_servers.sh`, `run_workflow.sh` — utilities for local debugging (optional for Cloud Run).
- `test_cloudrun.py` — simple client to submit a job and poll status.
- `outputs/` — sample request and example output (to be added separately).

### Prerequisites
- Google Cloud project with billing enabled.
- `gcloud` CLI installed and authenticated: `gcloud auth login`.
- Artifact Registry API enabled.
- Cloud Run API enabled.
- A GCS bucket for outputs (e.g., `scout-agent-outputs` in `us-central1`).

### Build and Push the Container
Replace values as needed (project, region, repo, image).

```bash
PROJECT_ID="your-gcp-project"
REGION="us-central1"
REPO="scout-agent"
IMAGE="scout-agent"

cd /home/ubuntu/ScoutAgent
sudo docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest .

gcloud auth configure-docker ${REGION}-docker.pkg.dev
sudo docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest
```

### Create the Output Bucket
```bash
gsutil mb -l us-central1 gs://scout-agent-outputs
```

### Deploy to Cloud Run (Console)
Deploy two services using the Google Cloud Console (recommended for simplicity).

#### 1) API Service (`scout-agent-api`)
- Image: `${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest`
- Port: `8080`
- Allow unauthenticated invocations (for public API)
- Command: `python`
- Args: `api_service.py`
- Environment variables:
  - `WORKER_SERVICE_URL` = `https://<your-worker-url>` (full URL to worker service)

#### 2) Worker Service (`scout-agent-worker`)
- Image: same as API
- Port: `8080`
- **Request timeout: `3600` (1 hour) - CRITICAL for long Reddit API calls**
- Allow unauthenticated (or keep authenticated and secure it; API can be updated to call with ID token)
- Command: `python`
- Args: `worker_service.py`
- Environment variables:
  - `GCS_BUCKET` = `scout-agent-outputs` (bucket name only; no `gs://`)
  - API keys (temporary as env vars; use Secret Manager for production):
    - `SCOUT_OPENAI_API_KEY`
    - `SCOUT_ANTHROPIC_API_KEY`
    - `SCOUT_GEMINI_API_KEY`
    - `SCOUT_DEEPSEEK_API_KEY`
    - `SCOUT_REDDIT_CLIENT_ID`
    - `SCOUT_REDDIT_CLIENT_SECRET`
    - `SCOUT_REDDIT_USER_AGENT`

#### Bucket Permissions
Grant the worker service account permission to write to the bucket:
- Bucket → Permissions → Grant access
- Principal: the service account for `scout-agent-worker`
- Role: Storage Object Admin (or Object Creator + Viewer)

### Running the System
#### Submit a Job (API)
```bash
API_URL="https://<your-api-url>"
curl -X POST "${API_URL}/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "target_market": "Knowledge management tools",
    "keywords": "bidirectional links,markdown sync",
    "subreddits": "PKMS,productivity,Evernote",
    "per_query_limit": 2
  }'
```
Response:
```json
{
  "job_id": "47bb3158-6977-4789-a3df-355a365557a6",
  "status": "pending",
  "created_at": "2025-09-30T12:34:56.000000",
  "estimated_duration": "5-15 minutes"
}
```

#### Check Job Status
```bash
curl "${API_URL}/jobs/<job_id>"
```
When complete:
```json
{
  "job_id": "<job_id>",
  "status": "completed",
  "created_at": "...",
  "completed_at": "...",
  "gcs_output_path": "gs://scout-agent-outputs/scout/jobs/<job_id>/"
}
```

### Testing Locally Against Cloud Run
Use the included script (update API URL inside the file):
```bash
python test_cloudrun.py
```
It will create a job, poll every 10 seconds, and print the final `gs://` path.

### Retrieve Outputs from GCS
- Console: Cloud Storage → your bucket → `scout/jobs/<job_id>/`
- CLI:
```bash
JOB_ID="<job_id>"
gsutil -m cp -r "gs://scout-agent-outputs/scout/jobs/${JOB_ID}" ./outputs/${JOB_ID}
```

### Troubleshooting
- **SSE timeout after 4 minutes (`httpcore.ReadTimeout`)**: 
  - **CAUSE**: Cloud Run's default 240s timeout is too short for Reddit API calls
  - **FIX**: Set `--timeout 3600` and `--request-timeout 3600` on worker service (already in deploy script)
  - Verify with: `gcloud run services describe scout-agent-worker --region us-central1 --format="value(spec.template.spec.timeoutSeconds)"`
- Job status = failed with error mentioning `storage.googleapis.com ... Not Found`:
  - Ensure `GCS_BUCKET` is set to the bucket name only (no URL), e.g. `scout-agent-outputs`.
- Job status = failed due to permissions:
  - Grant the worker service account write access to the bucket.
- LLM backend initialization errors:
  - Verify `SCOUT_*` API keys are present on the worker service.
- Long jobs timing out:
  - Worker now uses 1-hour timeout for SSE connections and Cloud Run requests.
- Worker cannot reach main app MCP servers:
  - MCP servers start automatically within the worker. Check worker logs for startup errors.

### Security Notes
- For production, move API keys to Secret Manager and reference them in Cloud Run.
- Restrict the API service or add auth if exposing publicly long term.

### License
MIT (or your preferred license).
