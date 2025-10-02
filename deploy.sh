#!/bin/bash

# Quick deployment script with correct project ID
# Run this to deploy all fixes to Cloud Run

set -e

PROJECT_ID="delvelabs-scout-agent"
REGION="us-central1"
IMAGE_NAME="scout-agent"

echo "🚀 Deploying ScoutAgent fixes to Cloud Run"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated. Run: gcloud auth login"
    exit 1
fi

# Set project
gcloud config set project $PROJECT_ID

echo "📦 Building container image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME .

echo ""
echo "🌐 Deploying API Service..."
gcloud run deploy scout-agent-api \
    --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
    --region $REGION \
    --allow-unauthenticated \
    --port 8080 \
    --memory 512Mi \
    --cpu 1 \
    --timeout 3600 \
    --set-env-vars "GCS_BUCKET=scout-agent-outputs" \
    --command python \
    --args api_service.py

API_URL=$(gcloud run services describe scout-agent-api --region $REGION --format="value(status.url)")

echo ""
echo "⚙️ Deploying Worker Service..."
gcloud run deploy scout-agent-worker \
    --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
    --region $REGION \
    --no-allow-unauthenticated \
    --port 8080 \
    --memory 4Gi \
    --cpu 2 \
    --max-instances 5 \
    --timeout 3600 \
    --set-env-vars "GCS_BUCKET=scout-agent-outputs,WORKER_SERVICE_URL=http://scout-agent-worker:8080,VERBOSE_MCP_LOGS=false" \
    --command python \
    --args worker_service.py

WORKER_URL=$(gcloud run services describe scout-agent-worker --region $REGION --format="value(status.url)")

echo ""
echo "✅ Deployment Complete!"
echo ""
echo "📊 Service URLs:"
echo "API: $API_URL"
echo "Worker: $WORKER_URL"
echo ""
echo "⚠️  IMPORTANT: Set environment variables for API keys:"
echo "gcloud run services update scout-agent-worker --region $REGION \\"
echo "  --set-env-vars 'SCOUT_OPENAI_API_KEY=sk-...,SCOUT_ANTHROPIC_API_KEY=sk-ant-...,SCOUT_REDDIT_CLIENT_ID=...,SCOUT_REDDIT_CLIENT_SECRET=...,SCOUT_REDDIT_USER_AGENT=scout-agent/1.0'"
echo ""
echo "🧪 Test with:"
echo "curl -X POST '$API_URL/jobs' -H 'Content-Type: application/json' -d '{\"target_market\":\"Knowledge management\",\"keywords\":\"bidirectional links\",\"subreddits\":\"PKMS\",\"per_query_limit\":2}'"
