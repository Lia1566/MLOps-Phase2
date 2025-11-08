# Docker Deployment Guide

## Overview

This guide explains how to build and run the Student Performance Prediction API using Docker.

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (usually comes with Docker Desktop)

## Quick Start

### 1. Build the Docker Image
```bash
docker-compose build
```

### 2. Run the Container
```bash
docker-compose up
```

### 3. Access the API

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Predictions**: http://localhost:8000/predict

### 4. Stop the Container
```bash
docker-compose down
```

---

## Manual Docker Commands

### Build Image
```bash
docker build -t student-performance-api:latest .
```

### Run Container
```bash
docker run -d \
  --name student-performance-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/config:/app/config:ro \
  student-performance-api:latest
```

### View Logs
```bash
docker logs -f student-performance-api
```

### Stop Container
```bash
docker stop student-performance-api
docker rm student-performance-api
```

---

## Testing the API

### Health Check
```bash
curl http://localhost:8000/health
```

### Make a Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Class_X_Percentage": 85.5,
    "Class_XII_Percentage": 78.0,
    "Study_Hours": 5.0,
    "Gender": "Male",
    "Caste": "General",
    "Coaching": "Yes",
    "Medium": "English"
  }'
```

---

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs student-performance-api

# Check if port 8000 is already in use
lsof -i :8000
```

### Model not found error

Ensure `models/pipeline_baseline.pkl` exists:
```bash
ls -la models/pipeline_baseline.pkl
```

### Rebuild from scratch
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```

---

## Production Deployment

### Environment Variables
```bash
docker run -d \
  -e API_ENV=production \
  -e LOG_LEVEL=info \
  -p 8000:8000 \
  student-performance-api:latest
```

### Resource Limits
```yaml
# In docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

---

## Image Information

- **Base Image**: Python 3.10-slim
- **Size**: ~500 MB
- **Port**: 8000
- **Health Check**: Enabled (30s interval)
