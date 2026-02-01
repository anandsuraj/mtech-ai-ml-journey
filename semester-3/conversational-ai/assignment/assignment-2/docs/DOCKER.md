# Docker Setup Guide

## Prerequisites

- Docker Desktop installed and running
- 8GB+ RAM available
- Internet connection

## Quick Start

```bash
# Build and run the system
docker-compose up --build

# Access the web UI at http://localhost:5000
```

## Docker Commands

### Build and Run

```bash
# Build image
docker-compose build

# Run containers
docker-compose up

# Run in detached mode (background)
docker-compose up -d

# Build and run in one command
docker-compose up --build
```

### Managing Containers

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# Stop containers
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

### Development

```bash
# Rebuild after code changes
docker-compose build

# Restart after changes
docker-compose restart

# Execute command in running container
docker-compose exec rag-system bash

# View container shell
docker exec -it hybrid-rag-system /bin/bash
```

## Data Persistence

Data is persisted using Docker volumes:

```yaml
volumes:
  - ./data:/app/data      # Wikipedia data and indices
  - ./reports:/app/reports # Evaluation reports
```

These directories persist on your host machine even when containers are removed.

## Customization

### Change Port

Edit `docker-compose.yml`:

```yaml
ports:
  - "8080:5000"  # Map host port 8080 to container port 5000
```

### Environment Variables

Add to `docker-compose.yml`:

```yaml
environment:
  - FLASK_ENV=production
  - MODEL_NAME=google/flan-t5-base
```

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 8G
```

## Troubleshooting

**Port already in use**:
```bash
# Change port in docker-compose.yml or stop the conflicting service
lsof -ti:5000 | xargs kill -9
```

**Build fails**:
```bash
# Clean Docker cache and rebuild
docker-compose build --no-cache
```

**Container exits immediately**:
```bash
# Check logs
docker-compose logs rag-system
```

**Out of memory**:
```bash
# Increase Docker memory in Docker Desktop settings
# Preferences > Resources > Memory
```

## Production Deployment

For production, update `docker-compose.yml`:

```yaml
services:
  rag-system:
    build: .
    restart: always
    environment:
      - FLASK_ENV=production
    deploy:
      resources:
        limits:
          memory: 8G
```

## Team Workflow

1. **First Setup**:
   ```bash
   git pull
   docker-compose up --build
   ```

2. **Daily Use**:
   ```bash
   docker-compose up
   ```

3. **After Updates**:
   ```bash
   git pull
   docker-compose up --build
   ```

4. **Clean Rebuild**:
   ```bash
   docker-compose down -v
   docker-compose up --build
   ```
