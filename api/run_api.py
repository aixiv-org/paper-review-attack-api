#!/usr/bin/env python3
"""
Prompt Injection Detection API Runner
Production-ready startup script
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run the API server"""
    
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    log_level = os.getenv("API_LOG_LEVEL", "info")
    
    print(f"🚀 Starting Prompt Injection Detection API")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Workers: {workers}")
    print(f"   Reload: {reload}")
    print(f"   Log Level: {log_level}")
    print()
    
    # Start server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,  # Can't use multiple workers with reload
        reload=reload,
        log_level=log_level,
        app_dir=str(Path(__file__).parent)
    )

if __name__ == "__main__":
    main()