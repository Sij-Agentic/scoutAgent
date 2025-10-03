# ScoutAgent Documentation Index

This folder contains detailed documentation for deploying and using ScoutAgent on Google Cloud Run.

## Quick Reference

### 📚 [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
**Start here!** Quick commands for deploying, testing, and using ScoutAgent.
- Deploy commands
- Submit jobs
- Watch progress
- Download results

---

## User Guides

### 👤 [USER_EXPERIENCE.md](USER_EXPERIENCE.md)
Complete user workflow from submission to download.
- What users see
- How to watch progress
- How to share with team
- Result access methods

### 🎯 [FINAL_SOLUTION.md](FINAL_SOLUTION.md)
Overview of the complete solution.
- Public progress tracking
- Predictable output locations
- Security model
- API endpoints

---

## Deployment Guides

### 🚀 [CLEAN_DEPLOYMENT.md](CLEAN_DEPLOYMENT.md)
Clean deployment with all fixes applied.
- Trafilatura version
- No use_cache
- All timeouts fixed

### 🔧 [DEPLOYMENT_FIXES.md](DEPLOYMENT_FIXES.md)
All deployment issues and fixes.
- Timeout fixes
- Manifest path resolution
- Fire-and-forget architecture

---

## Technical Documentation

### ⏱️ [TIMEOUT_FIXES_COMPLETE.md](TIMEOUT_FIXES_COMPLETE.md)
Complete timeout configuration across all layers.
- Cloud Run timeouts
- MCP SSE timeouts
- Process wait timeouts
- Log verbosity control

### 📊 [PROGRESS_TRACKING.md](PROGRESS_TRACKING.md)
Real-time progress tracking implementation.
- GCS progress logs
- Live updates
- 24-hour availability

### 🔓 [PUBLIC_PROGRESS_ACCESS.md](PUBLIC_PROGRESS_ACCESS.md)
Public progress URL implementation.
- Security model
- UUID-based access
- Sharing capabilities

### 📝 [STATUS_TRACKING_FIX.md](STATUS_TRACKING_FIX.md)
GCS-based status tracking.
- Status file format
- Completion detection
- Error handling

---

## Specific Fixes

### 🔄 [ASYNC_JOB_FIX.md](ASYNC_JOB_FIX.md)
Fire-and-forget job processing.
- Why synchronous failed
- Async architecture
- GCS polling

### ⏰ [CLOUDRUN_FIX.md](CLOUDRUN_FIX.md)
Initial Cloud Run timeout fix.
- 4-minute timeout issue
- SSE configuration

### 🌐 [WEB_EXTRACTOR_FIX.md](WEB_EXTRACTOR_FIX.md)
WebContentExtractor version mismatch fix.
- Trafilatura vs simple version
- use_cache removal

---

## Updates

### 📖 [README_UPDATES.md](README_UPDATES.md)
Summary of README changes.
- Quick start section
- User experience improvements
- Documentation structure

---

## Document Organization

### By Use Case

**I want to deploy ScoutAgent:**
1. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Commands
2. [CLEAN_DEPLOYMENT.md](CLEAN_DEPLOYMENT.md) - Full guide

**I want to understand how it works:**
1. [FINAL_SOLUTION.md](FINAL_SOLUTION.md) - Overview
2. [USER_EXPERIENCE.md](USER_EXPERIENCE.md) - User flow
3. [PROGRESS_TRACKING.md](PROGRESS_TRACKING.md) - Technical details

**I'm debugging an issue:**
1. [TIMEOUT_FIXES_COMPLETE.md](TIMEOUT_FIXES_COMPLETE.md) - Timeout issues
2. [STATUS_TRACKING_FIX.md](STATUS_TRACKING_FIX.md) - Status issues
3. [DEPLOYMENT_FIXES.md](DEPLOYMENT_FIXES.md) - All fixes

**I want to understand the architecture:**
1. [ASYNC_JOB_FIX.md](ASYNC_JOB_FIX.md) - Job processing
2. [PUBLIC_PROGRESS_ACCESS.md](PUBLIC_PROGRESS_ACCESS.md) - Progress tracking
3. [PROGRESS_TRACKING.md](PROGRESS_TRACKING.md) - Implementation

---

## All Documents

| Document | Purpose | Audience |
|----------|---------|----------|
| QUICK_REFERENCE.md | Quick commands | Everyone |
| USER_EXPERIENCE.md | User workflow | End users |
| FINAL_SOLUTION.md | Complete overview | Everyone |
| CLEAN_DEPLOYMENT.md | Deployment guide | Developers |
| DEPLOYMENT_FIXES.md | All fixes | Developers |
| TIMEOUT_FIXES_COMPLETE.md | Timeout configuration | DevOps |
| PROGRESS_TRACKING.md | Progress implementation | Developers |
| PUBLIC_PROGRESS_ACCESS.md | Public access | Developers |
| STATUS_TRACKING_FIX.md | Status tracking | Developers |
| ASYNC_JOB_FIX.md | Async architecture | Developers |
| CLOUDRUN_FIX.md | Initial timeout fix | DevOps |
| WEB_EXTRACTOR_FIX.md | Extractor fix | Developers |
| README_UPDATES.md | README changes | Documentation |

---

## Quick Links

- **Main README**: [../README.md](../README.md)
- **Test Script**: [../test_cloudrun.py](../test_cloudrun.py)
- **Deploy Script**: [../deploy.sh](../deploy.sh)
