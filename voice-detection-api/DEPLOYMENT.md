# Quick Deployment Guide

This guide will help you deploy the Voice Detection API with the web interface.

## Local Development

### 1. Start the Server

```bash
# Navigate to project directory
cd voice-detection-api

# Activate virtual environment
.venv\Scripts\activate

# Start the development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 3. Get Your API Key

```bash
# View your API key
type .env

# Or generate a new one
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Production Deployment

### Option 1: Deploy to Render.com (Recommended)

Your project already has `render.yaml` configured!

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add web interface"
   git push origin main
   ```

2. **Connect to Render**
   - Go to https://render.com
   - Create new Web Service
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml`

3. **Set Environment Variables**
   - Add `API_KEY` in Render dashboard
   - Generate secure key: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

4. **Deploy**
   - Render will automatically build and deploy
   - Your app will be available at `https://your-app.onrender.com`

### Option 2: Deploy to Heroku

1. **Create Heroku App**
   ```bash
   heroku create your-app-name
   ```

2. **Set Environment Variables**
   ```bash
   heroku config:set API_KEY=your-secure-api-key
   ```

3. **Create Procfile**
   ```
   web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

### Option 3: Deploy to VPS (DigitalOcean, AWS, etc.)

1. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv nginx
   ```

2. **Setup Application**
   ```bash
   cd /var/www/voice-detection-api
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Create Systemd Service**
   
   Create `/etc/systemd/system/voice-detection.service`:
   ```ini
   [Unit]
   Description=Voice Detection API
   After=network.target

   [Service]
   User=www-data
   WorkingDirectory=/var/www/voice-detection-api
   Environment="PATH=/var/www/voice-detection-api/.venv/bin"
   ExecStart=/var/www/voice-detection-api/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

   [Install]
   WantedBy=multi-user.target
   ```

4. **Configure Nginx**
   
   Create `/etc/nginx/sites-available/voice-detection`:
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       }
   }
   ```

5. **Enable and Start**
   ```bash
   sudo systemctl enable voice-detection
   sudo systemctl start voice-detection
   sudo ln -s /etc/nginx/sites-available/voice-detection /etc/nginx/sites-enabled/
   sudo systemctl restart nginx
   ```

6. **Setup SSL with Let's Encrypt**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d yourdomain.com
   ```

## Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Create docker-compose.yml

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_KEY=${API_KEY}
    volumes:
      - ./static:/app/static
    restart: unless-stopped
```

### 3. Deploy

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Environment Variables

Required environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `API_KEY` | Authentication key for API | `your-secure-random-key` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to Google Cloud credentials | `/path/to/credentials.json` |

## Post-Deployment Checklist

- [ ] API is accessible at your domain
- [ ] Web interface loads correctly
- [ ] Health check endpoint returns OK
- [ ] API documentation is accessible at `/docs`
- [ ] SSL certificate is installed (HTTPS)
- [ ] Environment variables are set
- [ ] API key authentication works
- [ ] File upload works correctly
- [ ] Results display properly
- [ ] Monitor logs for errors

## Monitoring

### Check Application Status

```bash
# Check if service is running
systemctl status voice-detection

# View logs
journalctl -u voice-detection -f

# Check Nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

### Health Check Endpoint

```bash
curl https://yourdomain.com/health
```

Expected response:
```json
{"status": "ok"}
```

## Scaling

### Horizontal Scaling

Use a load balancer with multiple instances:

```nginx
upstream voice_detection {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    location / {
        proxy_pass http://voice_detection;
    }
}
```

### Vertical Scaling

Increase worker processes:

```bash
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Application won't start

- Check Python version (requires 3.11+)
- Verify all dependencies are installed
- Check environment variables are set
- Review error logs

### 502 Bad Gateway

- Ensure application is running
- Check firewall settings
- Verify port is not blocked
- Check Nginx configuration

### Slow performance

- Increase worker processes
- Enable caching
- Use CDN for static files
- Optimize audio processing

## Security Recommendations

1. **Use strong API keys** (32+ characters)
2. **Enable HTTPS** (use Let's Encrypt)
3. **Set up firewall** (UFW, iptables)
4. **Regular updates** (system and dependencies)
5. **Monitor logs** for suspicious activity
6. **Rate limiting** (use Nginx or FastAPI middleware)
7. **File size limits** (prevent DoS attacks)

## Backup

### Backup Configuration

```bash
# Backup .env file
cp .env .env.backup

# Backup entire application
tar -czf voice-detection-backup.tar.gz /var/www/voice-detection-api
```

### Restore

```bash
# Restore from backup
tar -xzf voice-detection-backup.tar.gz -C /
```

## Support

For deployment issues:
1. Check application logs
2. Verify environment variables
3. Test API endpoints manually
4. Review Nginx/proxy configuration
5. Check firewall and security groups

---

**Happy Deploying! 🚀**
