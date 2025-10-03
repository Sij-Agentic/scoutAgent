# ScoutAgent

ScoutAgent is an AI-powered market research tool that analyzes Reddit conversations to identify pain points, validate market opportunities, and discover potential vendors in target markets.

## Quick Start - Testing the Workflow

### Run a Test Job
```bash
python3 test_cloudrun.py
```

### What You'll Get
When you submit a job, you'll immediately receive:

```
✅ Job created successfully!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 Job ID: abc-123-def-456-789
⏱️  Estimated Duration: 5-15 minutes

📊 Watch Progress (Public - Shareable!):
   https://scout-agent-api.run.app/jobs/abc-123/progress

📁 Final Output Location:
   gs://scout-agent-outputs/scout/jobs/abc-123/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 Tip: Share the progress URL with your team to watch together!
💡 Progress log available for 24 hours
```

### Watch Progress in Real-Time
```bash
# Copy the progress URL from above and watch live updates
watch -n 5 "curl -s https://scout-agent-api.run.app/jobs/YOUR_JOB_ID/progress | jq -r '.progress'"

# Or open in browser (shows JSON with progress log)
open https://scout-agent-api.run.app/jobs/YOUR_JOB_ID/progress
```

#### About the Test Script
- The script `test_cloudrun.py` prints the progress URL and final output location and then exits.
- It does not poll status or print failure messages while the job runs.
- Prefer watching the progress URL live and downloading results when complete.

**You'll see**:
- MCP server startup
- Workflow stages (scout_collect, gap_finder_collect, etc.)
- Progress updates as job runs
- Completion status

### Download Results
When the job completes, you'll see:
```
🎉 Job completed successfully!

📁 Download Results:
   curl "https://scout-agent-api.run.app/jobs/YOUR_JOB_ID/download" -o results.zip

📂 Or access directly from GCS:
   gsutil -m cp -r "gs://scout-agent-outputs/scout/jobs/YOUR_JOB_ID/" ./results/
```

### Share with Your Team
The progress URL is **public** (no authentication needed):
- Share it in Slack/email
- Team can watch progress together
- Available for 24 hours
- No GCS access required to watch progress

---

## Deployment Guide

### Overview

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
- `test_cloudrun.py` — simple client to submit a job and print the progress URL and final output (no status polling).
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

  - Progress logging (optional tuning):
    - `PROGRESS_FLUSH_INTERVAL` (seconds, default `5`) — how often to flush progress buffer to GCS
    - `PROGRESS_MAX_BUFFER` (lines, default `50`) — flush when buffer reaches this many lines
    - Notes: Progress logging is buffered and retried with backoff to avoid GCS 429s. You may see a slight (seconds) delay in progress updates.

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

#### Check Job Status (Optional)
```bash
curl "${API_URL}/jobs/<job_id>"
```
When complete:
```json
{
  "job_id": "<job_id>",
  "status": "completed",
  "created_at": "2025-10-03T00:00:00",
  "completed_at": "2025-10-03T00:15:00",
  "gcs_output_path": "gs://scout-agent-outputs/scout/jobs/abc-123/",
  "error_message": null
}
```

### Download Results
```bash
curl "https://scout-agent-api.run.app/jobs/{job_id}/download" -o results.zip
```

### Expected Progress Behavior
- The progress URL is public and updates throughout the run.
- You may see periodic "Heartbeat: job running" lines to indicate the job is active.
- Progress is buffered and flushed every few seconds to avoid rate limits; a small delay is expected.

---
### Retrieve Outputs

#### Option 1: Download via API (Recommended)
```bash
JOB_ID="<job_id>"
curl "https://scout-agent-api.run.app/jobs/${JOB_ID}/download" -o results.zip
unzip results.zip
```

#### Option 2: Access from GCS
#### Option 3: View in Console
Cloud Storage → `scout-agent-outputs` bucket → `scout/jobs/<job_id>/`

### Troubleshooting

#### Job Shows "Failed" But Still Running
- **Old issue**: Fixed! Now uses fire-and-forget architecture
- **Status updates**: Check progress URL for real-time status
- **Completion detection**: Status updates when job writes to GCS

#### Can't See Progress
**Check progress URL**:
```bash
curl "https://scout-agent-api.run.app/jobs/{job_id}/progress"
```

**If "not available"**: Job is starting, wait a few seconds

#### Job Actually Failed
**Check progress log for errors**:
```bash
curl "https://scout-agent-api.run.app/jobs/{job_id}/progress" | jq -r '.progress' | grep ERROR
```

**Check worker logs**:
```bash
gcloud run services logs read scout-agent-worker --region us-central1 | grep {job_id}
```

#### Common Issues
- **"No LLM backends initialized"**: API keys not set on worker service
- **Timeout after 1 hour**: Jobs longer than 1 hour need adjustment (reduce `per_query_limit`)
- **GCS permission errors**: Worker service account needs Storage Object Creator role
- **Progress log empty**: Job may have crashed during startup, check worker logs

---

## Documentation

Detailed documentation is available in the [`docs/`](docs/) folder:

- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Commands and quick start
- **[User Experience](docs/USER_EXPERIENCE.md)** - Complete user workflow
- **[Deployment Guide](docs/CLEAN_DEPLOYMENT.md)** - Full deployment instructions
- **[Progress Tracking](docs/PROGRESS_TRACKING.md)** - Real-time progress implementation
- **[All Fixes](docs/DEPLOYMENT_FIXES.md)** - Complete list of fixes applied

See [docs/INDEX.md](docs/INDEX.md) for a complete documentation index.

---

### License
MIT (or your preferred license).
