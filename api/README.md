# Prompt Injection Detection API

🚀 RESTful API for detecting prompt injection attacks in PDF documents.

## Quick Start

### 1. Install Dependencies
```bash
# Install API dependencies
pip install -r requirements.txt

# Or install from project root
pip install -r ../requirements.txt
```

### 2. Start the API Server
```bash
# Development mode (with auto-reload)
python run_api.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API
- **API Base URL**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Single File Detection
```bash
curl -X POST \
  "http://localhost:8000/detect/single" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "detector_type=ensemble" \
  -F "return_details=true"
```

### Batch File Detection
```bash
curl -X POST \
  "http://localhost:8000/detect/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "detector_type=ensemble"
```

### System Metrics
```bash
curl http://localhost:8000/metrics
```

## Testing

Run the test suite:
```bash
# Make sure API is running first
python test_api.py
```

## Configuration

### Environment Variables
```bash
export API_HOST="0.0.0.0"
export API_PORT="8000"
export API_WORKERS="4"
export API_LOG_LEVEL="info"
export API_RELOAD="false"
```

### Production Configuration
```bash
# Production mode
API_WORKERS=4 API_RELOAD=false python run_api.py
```

## Performance

### Expected Performance
- **Single File**: < 1 second response time
- **Batch Processing**: ~2-4 files per second
- **Throughput**: 100+ requests per minute
- **Memory Usage**: ~2GB per worker

### Scaling
- Use multiple workers: `API_WORKERS=4`
- Deploy behind load balancer for horizontal scaling
- Consider caching for frequently analyzed files

## Security

### File Handling
- Only PDF files are accepted
- Files are automatically cleaned up after processing
- Temporary files are stored securely
- No long-term file storage

### Rate Limiting
- Built-in protection against abuse
- Configurable request limits
- Automatic cleanup of resources

## API Response Examples

### Single File Detection Response
```json
{
  "file_name": "suspicious_document.pdf",
  "is_malicious": true,
  "risk_score": 0.85,
  "detection_count": 3,
  "detection_types": ["keyword_injection", "white_text"],
  "processing_time": 0.42,
  "timestamp": "2025-08-04T01:00:00Z",
  "details": null
}
```

### Batch Detection Response
```json
{
  "total_files": 5,
  "malicious_files": 2,
  "clean_files": 3,
  "processing_time": 1.23,
  "results": [...],
  "summary": {
    "malicious_rate": 0.4,
    "average_risk_score": 0.34,
    "detection_types": {
      "keyword_injection": 3,
      "metadata_injection": 1
    },
    "files_per_second": 4.07
  }
}
```

## Error Handling

### Common Error Codes
- `400 Bad Request`: Invalid file type or parameters
- `422 Unprocessable Entity`: Validation errors
- `429 Too Many Requests`: Rate limiting
- `500 Internal Server Error`: Processing errors

### Error Response Format
```json
{
  "detail": "Only PDF files are supported"
}
```

## Development

### Project Structure
```
api/
├── main.py              # FastAPI application
├── run_api.py           # Production runner
├── test_api.py          # API tests
├── requirements.txt     # API dependencies
└── README.md           # This file
```

### Adding New Endpoints
1. Define endpoint in `main.py`
2. Add Pydantic models for request/response
3. Implement business logic
4. Add tests in `test_api.py`
5. Update documentation

## Troubleshooting

### Common Issues

**API won't start**
- Check if port 8000 is available
- Verify all dependencies are installed
- Check project root configuration is correct

**Detection errors**
- Ensure PDF files are valid
- Check file size limits
- Verify detector models are loaded

**Performance issues**
- Increase worker count
- Check system resources
- Monitor memory usage

### Logs
```bash
# Check API logs
tail -f api.log

# Debug mode
API_LOG_LEVEL=debug python run_api.py
```

## Docker Deployment

### Build Docker Image
```bash
docker build -t prompt-detection-api .
```

### Run Container
```bash
docker run -p 8000:8000 prompt-detection-api
```

### Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_WORKERS=4
      - API_LOG_LEVEL=info
```

## Monitoring

### Health Checks
```bash
# Kubernetes readiness probe
curl -f http://localhost:8000/health || exit 1
```

### Metrics Collection
- Use `/metrics` endpoint for monitoring
- Track response times and error rates
- Monitor system resources

## Support

- **Documentation**: See `/docs` endpoint for interactive API docs
- **Issues**: Contact development team
- **Performance**: Monitor `/metrics` for system health