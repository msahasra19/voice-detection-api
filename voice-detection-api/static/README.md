# Voice Detection Web Interface

A beautiful, modern web interface for the Voice Detection API that allows users to upload audio files and get AI vs Human voice classification results.

## Features

✨ **Modern UI/UX**
- Stunning gradient design with glassmorphism effects
- Smooth animations and transitions
- Responsive layout for all devices
- Dark mode optimized

🎯 **Core Functionality**
- Drag & drop audio file upload
- Support for MP3, WAV, FLAC, M4A formats
- Real-time API status indicator
- Comprehensive results display

📊 **Results Display**
- AI vs Human classification
- Confidence score and level
- Language detection (Tamil, English, Hindi, Malayalam, Telugu)
- Deepfake risk assessment
- Audio quality metrics (SNR, clipping detection)
- Detailed explainability insights
- Segment-by-segment analysis timeline

🔒 **Security**
- API key authentication
- Secure local storage of API key
- Client-side file processing

## Quick Start

### 1. Start the API Server

From the `voice-detection-api` directory:

```bash
# Activate virtual environment
.venv\Scripts\activate

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:8000
```

### 3. Get Your API Key

Your API key is stored in the `.env` file. To view it:

```bash
type .env
```

Or generate a new one:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 4. Use the Interface

1. **Enter your API key** in the API Key field (it will be saved locally)
2. **Upload an audio file** by clicking or dragging & dropping
3. **Click "Analyze Audio"** to get results
4. **View comprehensive results** including classification, language, quality metrics, and insights

## File Structure

```
static/
├── index.html      # Main HTML structure
├── styles.css      # Modern CSS with animations and gradients
└── script.js       # JavaScript for API interaction and UI logic
```

## API Integration

The web interface communicates with the FastAPI backend using:

- **Endpoint**: `POST /predict`
- **Headers**: `X-API-Key: your-api-key`
- **Body**: JSON with base64 encoded audio data
- **Response**: Comprehensive voice analysis results

## Customization

### Change API URL

Edit `script.js` and modify:

```javascript
const API_BASE_URL = 'https://your-api-domain.com';
```

### Modify Colors

Edit `styles.css` and update CSS variables in `:root`:

```css
:root {
    --accent-purple: #667eea;
    --accent-pink: #f093fb;
    /* ... more colors */
}
```

### Add New Features

The code is well-structured and commented:
- **HTML**: Semantic structure with clear sections
- **CSS**: Organized with CSS variables and utility classes
- **JavaScript**: Modular functions with clear responsibilities

## Browser Support

- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Opera

## Deployment

### Deploy to Production

1. **Build for production** (if using a bundler)
2. **Set the correct API URL** in `script.js`
3. **Serve static files** through your web server (Nginx, Apache, etc.)
4. **Enable HTTPS** for secure API key transmission

### Example Nginx Configuration

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        root /path/to/voice-detection-api/static;
        try_files $uri $uri/ /index.html;
    }

    location /predict {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /health {
        proxy_pass http://localhost:8000;
    }
}
```

### Deploy with FastAPI (Current Setup)

The current setup serves the web interface directly from FastAPI:

```python
# Already configured in app/main.py
app.mount("/static", StaticFiles(directory="static"))

@app.get("/")
async def root():
    return FileResponse("static/index.html")
```

This means you can deploy the entire application as one unit.

## Troubleshooting

### API Status shows "Offline"

- Check if the API server is running
- Verify the API URL in `script.js`
- Check browser console for CORS errors

### "Invalid or missing API key" error

- Ensure you've entered the correct API key
- Check that the API key matches the one in `.env`
- Verify the API key is being sent in the `X-API-Key` header

### File upload fails

- Check file format (must be audio)
- Verify file size isn't too large
- Check browser console for errors

### Results not displaying

- Open browser developer tools (F12)
- Check Console tab for JavaScript errors
- Verify API response format matches expected schema

## Performance Tips

1. **Optimize audio files** before upload (compress if needed)
2. **Use modern browsers** for best performance
3. **Enable caching** for static assets in production
4. **Use CDN** for serving static files at scale

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use HTTPS** in production
3. **Implement rate limiting** on the API
4. **Validate file types** on both client and server
5. **Set maximum file size limits**

## Future Enhancements

Potential features to add:

- [ ] Audio playback with waveform visualization
- [ ] Batch file processing
- [ ] Export results as PDF/CSV
- [ ] User accounts and history
- [ ] Real-time audio recording
- [ ] Comparison mode for multiple files
- [ ] Advanced filtering and search
- [ ] Dark/Light theme toggle

## License

This web interface is part of the Voice Detection API project.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review browser console for errors
3. Verify API server is running
4. Check API documentation at `/docs`

---

**Powered by Advanced AI Voice Detection Technology** 🚀
