#!/bin/bash

# ScoutAgent Cloud Run Deployment Script
# Deploys both API and Worker services to Google Cloud Run

set -e

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"delvelabs-scout-agent"}
REGION=${GOOGLE_CLOUD_REGION:-"us-central1"}
SERVICE_NAME_API="scout-agent-api"
SERVICE_NAME_WORKER="scout-agent-worker"
IMAGE_NAME="scout-agent"
BUCKET_NAME=${GCS_BUCKET:-"scout-agent-outputs"}

echo "🚀 Deploying ScoutAgent to Google Cloud Run"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Bucket: $BUCKET_NAME"

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run: gcloud auth login"
    exit 1
fi

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "📋 Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Create GCS bucket if it doesn't exist
echo "🪣 Creating GCS bucket..."
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME/ || echo "Bucket already exists"

# Build and push Docker image
echo "🐳 Building and pushing Docker image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME .

# Deploy API Service
echo "🌐 Deploying API Service..."
gcloud run deploy $SERVICE_NAME_API \
    --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 8080 \
    --memory 512Mi \
    --cpu 1 \
    --max-instances 10 \
    --timeout 3600 \
    --set-env-vars "GCS_BUCKET=$BUCKET_NAME" \
    --command "python" \
    --args "api_service.py"

# Get API service URL
API_URL=$(gcloud run services describe $SERVICE_NAME_API --region $REGION --format="value(status.url)")

# Deploy Worker Service
echo "⚙️ Deploying Worker Service..."
gcloud run deploy $SERVICE_NAME_WORKER \
    --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --no-allow-unauthenticated \
    --port 8080 \
    --memory 4Gi \
    --cpu 2 \
    --max-instances 5 \
    --timeout 3600 \
    --request-timeout 3600 \
    --set-env-vars "GCS_BUCKET=$BUCKET_NAME,WORKER_SERVICE_URL=http://$SERVICE_NAME_WORKER:8080" \
    --command "python" \
    --args "worker_service.py"

# Get Worker service URL
WORKER_URL=$(gcloud run services describe $SERVICE_NAME_WORKER --region $REGION --format="value(status.url)")

echo ""
echo "✅ Deployment Complete!"
echo ""
echo "📊 Service URLs:"
echo "API Service: $API_URL"
echo "Worker Service: $WORKER_URL"
echo ""
echo "🧪 Test the API:"
echo "curl -X POST '$API_URL/jobs' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"target_market\": \"Knowledge management tools\", \"keywords\": \"bidirectional links,markdown sync\", \"subreddits\": \"PKMS,productivity,Evernote\", \"per_query_limit\": 2}'"
echo ""
echo "📁 Outputs will be stored in: gs://$BUCKET_NAME/scout/jobs/{job_id}/"
echo ""
echo "🔧 To add environment variables (API keys), run:"
echo "gcloud run services update $SERVICE_NAME_WORKER --region $REGION --set-env-vars 'SCOUT_OPENAI_API_KEY=your_key,SCOUT_ANTHROPIC_API_KEY=your_key'"
