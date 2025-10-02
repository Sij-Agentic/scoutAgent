#!/bin/bash

# Quick fix script to update Cloud Run timeout settings
# Run this to fix the 4-minute SSE timeout issue without full redeployment

set -e

PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"your-project-id"}
REGION=${GOOGLE_CLOUD_REGION:-"us-central1"}
SERVICE_NAME_API="scout-agent-api"
SERVICE_NAME_WORKER="scout-agent-worker"

echo "🔧 Fixing Cloud Run timeout settings for SSE connections"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run: gcloud auth login"
    exit 1
fi

# Set project
gcloud config set project $PROJECT_ID

echo "📊 Current timeout settings:"
echo "API Service:"
gcloud run services describe $SERVICE_NAME_API --region $REGION --format="value(spec.template.spec.timeoutSeconds)" 2>/dev/null || echo "  Not found"
echo "Worker Service:"
gcloud run services describe $SERVICE_NAME_WORKER --region $REGION --format="value(spec.template.spec.timeoutSeconds)" 2>/dev/null || echo "  Not found"
echo ""

# Update API Service timeout
echo "⏱️  Updating API Service timeout to 3600s (1 hour)..."
gcloud run services update $SERVICE_NAME_API \
    --region $REGION \
    --timeout 3600 \
    --quiet

echo "✅ API Service timeout updated"

# Update Worker Service timeout
echo "⏱️  Updating Worker Service timeout to 3600s (1 hour)..."
gcloud run services update $SERVICE_NAME_WORKER \
    --region $REGION \
    --timeout 3600 \
    --request-timeout 3600 \
    --quiet

echo "✅ Worker Service timeout updated"

echo ""
echo "📊 New timeout settings:"
echo "API Service:"
gcloud run services describe $SERVICE_NAME_API --region $REGION --format="value(spec.template.spec.timeoutSeconds)"
echo "Worker Service:"
gcloud run services describe $SERVICE_NAME_WORKER --region $REGION --format="value(spec.template.spec.timeoutSeconds)"
echo ""

echo "✅ Timeout fix complete!"
echo ""
echo "🧪 Test the fix by submitting a new job:"
API_URL=$(gcloud run services describe $SERVICE_NAME_API --region $REGION --format="value(status.url)")
echo "curl -X POST '$API_URL/jobs' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"target_market\": \"Knowledge management tools\", \"keywords\": \"bidirectional links\", \"subreddits\": \"PKMS\", \"per_query_limit\": 2}'"
echo ""
