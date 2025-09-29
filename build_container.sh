#!/bin/bash

# ScoutAgent Container Build Script
set -e

# Configuration
IMAGE_NAME="scout-agent"
TAG="latest"
CONTAINER_NAME="scout-agent-test"

echo "🐳 Building ScoutAgent Docker container..."
echo "Image: ${IMAGE_NAME}:${TAG}"
echo "=" * 50

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t ${IMAGE_NAME}:${TAG} .

echo "✅ Docker image built successfully!"
echo "Image size:"
docker images ${IMAGE_NAME}:${TAG} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo ""
echo "🧪 Testing container..."
echo "=" * 50

# Test the container
echo "Running container test..."
docker run --rm \
    --name ${CONTAINER_NAME} \
    -v $(pwd)/test_container.py:/app/test_container.py \
    -v $(pwd)/test_mcp_servers.py:/app/test_mcp_servers.py \
    ${IMAGE_NAME}:${TAG} \
    python test_container.py

echo ""
echo "🧪 Testing MCP servers..."
echo "Running MCP server test (will start servers and test for 10 seconds)..."
timeout 15 docker run --rm \
    --name ${CONTAINER_NAME}-mcp \
    -v $(pwd)/test_mcp_servers.py:/app/test_mcp_servers.py \
    ${IMAGE_NAME}:${TAG} \
    python test_mcp_servers.py || true

echo ""
echo "🎉 Container build and test completed successfully!"
echo ""
echo "To run ScoutAgent manually:"
echo "docker run --rm ${IMAGE_NAME}:${TAG} python -m scout_agent.main --help"
echo ""
echo "To run with your example command:"
echo "docker run --rm ${IMAGE_NAME}:${TAG} python -m scout_agent.main \\"
echo "  --target-market \"Knowledge management tools\" \\"
echo "  --keywords \"bidirectional links,markdown sync,PDF annotation,template friction,backlink noise\" \\"
echo "  --subreddits \"PKMS,productivity,Evernote\" \\"
echo "  --per-query-limit 2"
