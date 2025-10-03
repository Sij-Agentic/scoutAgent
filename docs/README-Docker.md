# ScoutAgent Docker Container

This directory contains the Docker setup for containerizing the ScoutAgent system.

## Files

- `Dockerfile` - Multi-stage Docker build configuration
- `requirements-docker.txt` - Cleaned requirements file for container builds
- `.dockerignore` - Files to exclude from Docker build context
- `build_container.sh` - Build and test script
- `test_container.py` - Container functionality test script

## Quick Start

### Build the Container

```bash
# Make the build script executable
chmod +x build_container.sh

# Build and test the container
./build_container.sh
```

### Manual Build

```bash
# Build the Docker image
docker build -t scout-agent:latest .

# Test the container
docker run --rm scout-agent:latest python test_container.py
```

### Run ScoutAgent

```bash
# Show help (includes MCP server startup)
docker run --rm scout-agent:latest ./start_services.sh --help

# Run with your parameters (MCP servers start automatically)
docker run --rm scout-agent:latest ./start_services.sh \
  --target-market "Knowledge management tools" \
  --keywords "bidirectional links,markdown sync,PDF annotation,template friction,backlink noise" \
  --subreddits "PKMS,productivity,Evernote" \
  --per-query-limit 2
```

### Test MCP Servers

```bash
# Test that MCP servers can start and run
docker run --rm scout-agent:latest python test_mcp_servers.py
```

## Container Features

### Multi-Stage Build
- **Builder stage**: Compiles dependencies with build tools
- **Runtime stage**: Minimal runtime with only necessary libraries
- **Size optimization**: Removes build dependencies from final image

### MCP Server Integration
- **Automatic startup**: All required MCP servers start automatically
- **Service orchestration**: `start_services.sh` manages server lifecycle
- **Health monitoring**: Built-in service health checks
- **Graceful shutdown**: Proper cleanup on container exit

### Security
- Non-root user (`scoutagent`) for running the application
- Minimal attack surface with slim base image
- No unnecessary packages or tools

### Performance
- Virtual environment for clean dependency management
- Optimized layer caching for faster rebuilds
- Multi-threaded dependency installation

## Environment Variables

The container sets these default environment variables:

- `PYTHONPATH=/app`
- `PYTHONUNBUFFERED=1`
- `SCOUT_LLM_DEFAULT_BACKEND=deepseek`
- `SCOUT_LLM_DEFAULT_MODEL=deepseek-chat`

## Output

The container creates these directories:
- `/app/output` - For ScoutAgent output files
- `/app/logs` - For log files
- `/app/temp` - For temporary files

## Next Steps

This container is ready for:
1. **Local testing** - Run ScoutAgent in isolated environment
2. **Cloud deployment** - Deploy to AWS Fargate, Google Cloud Run, etc.
3. **API integration** - Add FastAPI wrapper for HTTP endpoints
4. **Job queue** - Integrate with Celery/Redis for async processing

## Troubleshooting

### Build Issues
- Ensure Docker has enough memory (4GB+ recommended)
- Check that all requirements files exist
- Verify Python version compatibility

### Runtime Issues
- Check container logs: `docker logs <container_id>`
- Verify environment variables are set correctly
- Ensure output directories have proper permissions

### Performance Issues
- Monitor container resource usage
- Consider increasing memory limits for large workloads
- Use multi-stage builds to reduce image size
