# Deployment Guide

Deployment options for the Dashboard Ceres Wealth.

## Table of Contents

1. [Docker Deployment](#docker-deployment)
2. [Cloud Deployment](#cloud-deployment)
3. [Production Considerations](#production-considerations)
4. [Monitoring & Logging](#monitoring--logging)

## Docker Deployment

### Production Build

Optimize the Docker build for production:

```dockerfile
# Build stage
FROM python:3.12.11-slim-bookworm as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12.11-slim-bookworm

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core fonts-liberation \
    libasound2 libnss3 libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
```

### Running Production Container

```bash
# Production run with environment file
docker run -d \
  --name ceres-dashboard \
  --restart unless-stopped \
  -p 8080:8080 \
  --env-file .env \
  -v ceres-data:/app/databases \
  ceres-dashboard:latest
```

### Health Checks

Add to `docker-compose.yml`:

```yaml
services:
  dashboard:
    build: .
    ports:
      - "8080:8080"
    environment:
      - PASSWORD=${PASSWORD}
    volumes:
      - ceres-data:/app/databases
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped

volumes:
  ceres-data:
```

## Cloud Deployment

### AWS Deployment

#### ECS (Elastic Container Service)

1. **Push to ECR**:
```bash
aws ecr get-login-password | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker tag ceres-dashboard:latest <account-id>.dkr.ecr.<region>.amazonaws.com/ceres-dashboard:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/ceres-dashboard:latest
```

2. **Create ECS Task Definition**:
```json
{
  "family": "ceres-dashboard",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [{
    "name": "ceres-dashboard",
    "image": "<account-id>.dkr.ecr.<region>.amazonaws.com/ceres-dashboard:latest",
    "portMappings": [{"containerPort": 8080, "protocol": "tcp"}],
    "environment": [{"name": "PASSWORD", "value": "<secure-password>"}],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/ceres-dashboard",
        "awslogs-region": "<region>",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }]
}
```

3. **Deploy with ECS Service** or use AWS Copilot:
```bash
copilot init --app ceres --name dashboard --type 'Load Balanced Web Service' --dockerfile './Dockerfile'
copilot deploy
```

#### EC2 Deployment

For persistent EC2 instance:

```bash
# User data script for EC2
#!/bin/bash
yum update -y
yum install -y docker
service docker start
docker pull <ecr-uri>/ceres-dashboard:latest
docker run -d \
  --name ceres-dashboard \
  -p 80:8080 \
  -e PASSWORD=<secure-password> \
  -v /data:/app/databases \
  <ecr-uri>/ceres-dashboard:latest
```

### Google Cloud Platform

#### Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/ceres-dashboard

# Deploy to Cloud Run
gcloud run deploy ceres-dashboard \
  --image gcr.io/PROJECT_ID/ceres-dashboard \
  --platform managed \
  --region us-central1 \
  --port 8080 \
  --set-env-vars PASSWORD=<secure-password> \
  --memory 2Gi \
  --cpu 1 \
  --allow-unauthenticated
```

**Note**: Cloud Run has request timeout limits (3600s max). For long-running operations, consider Cloud Run jobs or GKE.

#### GKE (Google Kubernetes Engine)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ceres-dashboard
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ceres-dashboard
  template:
    metadata:
      labels:
        app: ceres-dashboard
    spec:
      containers:
      - name: ceres-dashboard
        image: gcr.io/PROJECT_ID/ceres-dashboard:latest
        ports:
        - containerPort: 8080
        env:
        - name: PASSWORD
          valueFrom:
            secretKeyRef:
              name: ceres-secrets
              key: password
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: ceres-dashboard-service
spec:
  selector:
    app: ceres-dashboard
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
```

### Azure Deployment

#### Azure Container Instances

```bash
az container create \
  --resource-group ceres-rg \
  --name ceres-dashboard \
  --image <registry>/ceres-dashboard:latest \
  --cpu 1 \
  --memory 2 \
  --ports 8080 \
  --environment-variables PASSWORD=<secure-password> \
  --dns-name-label ceres-dashboard
```

## Production Considerations

### Security

1. **HTTPS**: Always use HTTPS in production
   - Use a reverse proxy (nginx, traefik) or cloud load balancer
   - Configure SSL certificates

2. **Secrets Management**:
   - Never commit passwords to version control
   - Use cloud secret managers (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault)
   - Rotate passwords regularly

3. **Network Security**:
   - Restrict access with IP whitelisting
   - Use VPN or private networks where possible
   - Enable security groups/firewall rules

### Performance

1. **Resource Allocation**:
   - Minimum: 1 CPU, 2GB RAM
   - Recommended: 2 CPU, 4GB RAM for concurrent users

2. **Caching**:
   - Enable Streamlit caching with `@st.cache_data`
   - Consider Redis for distributed caching (multi-instance deployments)

3. **Database**:
   - SQLite is sufficient for single-instance deployments
   - For high availability, migrate to PostgreSQL or cloud databases
   - Implement regular backups

### Scalability

**Single Instance**: Suitable for 1-10 concurrent users

**Multiple Instances**: Requires:
- Shared database (RDS, Cloud SQL, etc.)
- Session affinity or shared session store
- Load balancer

**Architecture for Scale**:
```
Load Balancer
    ├── Instance 1 (Streamlit)
    ├── Instance 2 (Streamlit)
    └── Instance N (Streamlit)
           ↓
      Shared Database (PostgreSQL/RDS)
```

### Data Persistence

**Docker Volumes**:
```bash
# Create named volume
docker volume create ceres-data

# Backup
docker run --rm -v ceres-data:/data -v $(pwd):/backup alpine tar czf /backup/backup.tar.gz -C /data .

# Restore
docker run --rm -v ceres-data:/data -v $(pwd):/backup alpine sh -c "cd /data && tar xzf /backup/backup.tar.gz"
```

**Cloud Storage**:
- Mount cloud storage buckets for data files
- Use cloud database services for relational data

### Backup Strategy

1. **Database Backups**:
   - Daily automated backups
   - Store backups in separate region/account
   - Test restore procedures

2. **Configuration Backups**:
   - Infrastructure as Code (Terraform, CloudFormation)
   - Version control for all configurations

## Monitoring & Logging

### Application Logging

Streamlit logs to stdout/stderr by default. Capture with:

```bash
# Docker logging
docker logs -f ceres-dashboard

# Or redirect to file
docker run ... > app.log 2>&1
```

### Health Monitoring

Streamlit provides a health endpoint at `/_stcore/health`.

**Monitoring Setup**:

```python
# Add to monitoring.py or health check endpoint
import requests

def check_health():
    try:
        response = requests.get('http://localhost:8080/_stcore/health')
        return response.status_code == 200
    except:
        return False
```

### CloudWatch (AWS)

```bash
# Install CloudWatch agent on EC2
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i amazon-cloudwatch-agent.deb

# Configure for Docker logs
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard
```

### Prometheus/Grafana (Kubernetes)

```yaml
# ServiceMonitor for Prometheus
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ceres-dashboard
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      app: ceres-dashboard
  endpoints:
  - port: http
    path: /_stcore/health
```

### Alerting

Set up alerts for:
- Container restarts
- High memory/CPU usage
- Application errors
- Health check failures

Example CloudWatch alarm:
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name ceres-dashboard-health \
  --alarm-description "Dashboard health check failed" \
  --metric-name HealthCheckStatus \
  --namespace Custom/Ceres \
  --statistic Average \
  --period 300 \
  --threshold 1 \
  --comparison-operator LessThanThreshold \
  --evaluation-periods 2
```

## CI/CD Pipeline

### GitHub Actions Example

```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t ceres-dashboard:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        echo ${{ secrets.REGISTRY_PASSWORD }} | docker login -u ${{ secrets.REGISTRY_USERNAME }} --password-stdin
        docker push ceres-dashboard:${{ github.sha }}
    
    - name: Deploy to production
      run: |
        # Update ECS service or run deployment script
        aws ecs update-service --cluster ceres --service dashboard --force-new-deployment
```

## Rollback Strategy

Always maintain the previous version for quick rollback:

```bash
# Tag current version
docker tag ceres-dashboard:latest ceres-dashboard:backup-$(date +%Y%m%d)

# Deploy new version
docker run -d --name ceres-dashboard-new ceres-dashboard:new-version

# If issues, rollback
docker stop ceres-dashboard-new
docker start ceres-dashboard-previous
```

---

**Note**: This is a single-tenant internal tool. For enterprise multi-tenant deployments, significant architecture changes would be needed (user management, database per tenant, etc.).
