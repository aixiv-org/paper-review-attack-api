#!/usr/bin/env python3
"""
Prompt Injection Detection API
RESTful API service for detecting prompt injection attacks in PDF documents

Date: 2025-08-04
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import detection modules
from src.detector import PromptInjectionDetector, EnsembleDetector
from src.utils import setup_logging, load_config, validate_pdf

# Configure logging
logger = setup_logging("INFO")

# Initialize FastAPI app
app = FastAPI(
    title="Prompt Injection Detection API",
    description="RESTful API for detecting prompt injection attacks in PDF documents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
config = None
detector = None
ensemble_detector = None
executor = ThreadPoolExecutor(max_workers=4)

# Pydantic models for API
class DetectionRequest(BaseModel):
    """Request model for detection"""
    detector_type: str = Field(default="ensemble", description="Detector type: 'standard' or 'ensemble'")
    threshold: Optional[float] = Field(default=None, description="Custom risk threshold")
    return_details: bool = Field(default=False, description="Include detailed detection results")

class DetectionResult(BaseModel):
    """Response model for single file detection"""
    file_name: str
    is_malicious: bool
    risk_score: float
    detection_count: int
    detection_types: List[str]
    processing_time: float
    timestamp: str
    details: Optional[Dict[str, Any]] = None

class BatchDetectionResult(BaseModel):
    """Response model for batch detection"""
    total_files: int
    malicious_files: int
    clean_files: int
    processing_time: float
    results: List[DetectionResult]
    summary: Dict[str, Any]

class HealthStatus(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    detectors_loaded: bool
    uptime: float

class APIMetrics(BaseModel):
    """API metrics response"""
    total_requests: int
    successful_detections: int
    failed_detections: int
    average_processing_time: float
    detector_stats: Dict[str, Any]

# Global metrics tracking
api_metrics = {
    "total_requests": 0,
    "successful_detections": 0,
    "failed_detections": 0,
    "processing_times": [],
    "start_time": datetime.now()
}

@app.on_event("startup")
async def startup_event():
    """Initialize detectors on startup"""
    global config, detector, ensemble_detector
    
    try:
        logger.info("🚀 Starting Prompt Injection Detection API...")
        
        # Load configuration
        config_path = project_root / "config" / "config.yaml"
        config = load_config(str(config_path))
        logger.info("✅ Configuration loaded successfully")
        
        # Initialize detectors
        detector = PromptInjectionDetector(config)
        ensemble_detector = EnsembleDetector(config)
        logger.info("✅ Detectors initialized successfully")
        
        logger.info("🎉 API startup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Prompt Injection Detection API...")
    executor.shutdown(wait=True)

def get_detector(detector_type: str = "ensemble"):
    """Get detector instance based on type"""
    if detector_type == "standard":
        return detector
    elif detector_type == "ensemble":
        return ensemble_detector
    else:
        raise HTTPException(status_code=400, detail=f"Invalid detector type: {detector_type}")

async def save_uploaded_file(upload_file: UploadFile) -> tuple[str, str]:
    """Save uploaded file to temporary location"""
    if not upload_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create temporary file
    temp_id = str(uuid.uuid4())
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"api_upload_{temp_id}.pdf")
    
    try:
        # Save file
        with open(temp_path, "wb") as buffer:
            content = await upload_file.read()
            buffer.write(content)
        
        # Validate PDF
        if not validate_pdf(temp_path):
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="Invalid PDF file")
        
        return temp_path, upload_file.filename
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {str(e)}")

def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

def run_detection(detector_instance, file_path: str, threshold: Optional[float] = None) -> Dict[str, Any]:
    """Run detection in thread pool"""
    try:
        start_time = datetime.now()
        
        # Override threshold if provided
        if threshold is not None:
            original_threshold = detector_instance.detection_config['thresholds']['risk_score']
            detector_instance.detection_config['thresholds']['risk_score'] = threshold
        
        # Run detection
        result = detector_instance.detect_injection(file_path)
        
        # Restore original threshold
        if threshold is not None:
            detector_instance.detection_config['thresholds']['risk_score'] = original_threshold
        
        # Add processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        result['processing_time'] = processing_time
        result['timestamp'] = datetime.now().isoformat()
        
        return result
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Prompt Injection Detection API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "detect_single": "/detect/single",
            "detect_batch": "/detect/batch"
        }
    }

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - api_metrics["start_time"]).total_seconds()
    
    return HealthStatus(
        status="healthy" if (detector is not None and ensemble_detector is not None) else "unhealthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        detectors_loaded=detector is not None and ensemble_detector is not None,
        uptime=uptime
    )

@app.get("/metrics", response_model=APIMetrics)
async def get_metrics():
    """Get API metrics"""
    avg_processing_time = (
        sum(api_metrics["processing_times"]) / len(api_metrics["processing_times"])
        if api_metrics["processing_times"] else 0.0
    )
    
    return APIMetrics(
        total_requests=api_metrics["total_requests"],
        successful_detections=api_metrics["successful_detections"],
        failed_detections=api_metrics["failed_detections"],
        average_processing_time=avg_processing_time,
        detector_stats={
            "standard_detector_available": detector is not None,
            "ensemble_detector_available": ensemble_detector is not None,
            "config_loaded": config is not None
        }
    )

@app.post("/detect/single", response_model=DetectionResult)
async def detect_single_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    detector_type: str = Query(default="ensemble", description="Detector type: 'standard' or 'ensemble'"),
    threshold: Optional[float] = Query(default=None, description="Custom risk threshold"),
    return_details: bool = Query(default=False, description="Include detailed detection results")
):
    """
    Detect prompt injection in a single PDF file
    
    Args:
        file: PDF file to analyze
        detector_type: Type of detector to use ('standard' or 'ensemble')
        threshold: Custom risk threshold (optional)
        return_details: Whether to include detailed results
    
    Returns:
        DetectionResult with analysis results
    """
    api_metrics["total_requests"] += 1
    temp_path = None
    
    try:
        # Save uploaded file
        temp_path, original_filename = await save_uploaded_file(file)
        
        # Get detector
        detector_instance = get_detector(detector_type)
        
        # Run detection in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            run_detection, 
            detector_instance, 
            temp_path, 
            threshold
        )
        
        # Update metrics
        api_metrics["successful_detections"] += 1
        api_metrics["processing_times"].append(result['processing_time'])
        
        # Keep only last 1000 processing times
        if len(api_metrics["processing_times"]) > 1000:
            api_metrics["processing_times"] = api_metrics["processing_times"][-1000:]
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_path)
        
        # Format response
        detection_result = DetectionResult(
            file_name=original_filename,
            is_malicious=result['is_malicious'],
            risk_score=result['risk_score'],
            detection_count=result['detection_count'],
            detection_types=result.get('detection_types', []),
            processing_time=result['processing_time'],
            timestamp=result['timestamp'],
            details=result if return_details else None
        )
        
        logger.info(f"Single file detection completed: {original_filename} - {'MALICIOUS' if result['is_malicious'] else 'CLEAN'}")
        
        return detection_result
        
    except HTTPException:
        api_metrics["failed_detections"] += 1
        if temp_path:
            cleanup_temp_file(temp_path)
        raise
    except Exception as e:
        api_metrics["failed_detections"] += 1
        if temp_path:
            cleanup_temp_file(temp_path)
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect/batch", response_model=BatchDetectionResult)
async def detect_batch_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    detector_type: str = Query(default="ensemble", description="Detector type: 'standard' or 'ensemble'"),
    threshold: Optional[float] = Query(default=None, description="Custom risk threshold"),
    return_details: bool = Query(default=False, description="Include detailed detection results"),
    max_files: int = Query(default=10, description="Maximum number of files to process")
):
    """
    Detect prompt injection in multiple PDF files
    
    Args:
        files: List of PDF files to analyze
        detector_type: Type of detector to use ('standard' or 'ensemble')
        threshold: Custom risk threshold (optional)
        return_details: Whether to include detailed results
        max_files: Maximum number of files to process
    
    Returns:
        BatchDetectionResult with analysis results for all files
    """
    api_metrics["total_requests"] += 1
    temp_paths = []
    start_time = datetime.now()
    
    try:
        # Validate file count
        if len(files) > max_files:
            raise HTTPException(
                status_code=400, 
                detail=f"Too many files. Maximum allowed: {max_files}, received: {len(files)}"
            )
        
        # Save all uploaded files
        file_info = []
        for file in files:
            temp_path, original_filename = await save_uploaded_file(file)
            temp_paths.append(temp_path)
            file_info.append((temp_path, original_filename))
        
        # Get detector
        detector_instance = get_detector(detector_type)
        
        # Run detections in parallel
        loop = asyncio.get_event_loop()
        detection_tasks = [
            loop.run_in_executor(
                executor, 
                run_detection, 
                detector_instance, 
                temp_path, 
                threshold
            ) for temp_path, _ in file_info
        ]
        
        detection_results = await asyncio.gather(*detection_tasks)
        
        # Process results
        results = []
        malicious_count = 0
        
        for i, result in enumerate(detection_results):
            original_filename = file_info[i][1]
            
            if result['is_malicious']:
                malicious_count += 1
            
            detection_result = DetectionResult(
                file_name=original_filename,
                is_malicious=result['is_malicious'],
                risk_score=result['risk_score'],
                detection_count=result['detection_count'],
                detection_types=result.get('detection_types', []),
                processing_time=result['processing_time'],
                timestamp=result['timestamp'],
                details=result if return_details else None
            )
            results.append(detection_result)
        
        # Calculate summary
        total_processing_time = (datetime.now() - start_time).total_seconds()
        clean_count = len(files) - malicious_count
        
        # Update metrics
        api_metrics["successful_detections"] += len(files)
        for result in detection_results:
            api_metrics["processing_times"].append(result['processing_time'])
        
        # Keep only last 1000 processing times
        if len(api_metrics["processing_times"]) > 1000:
            api_metrics["processing_times"] = api_metrics["processing_times"][-1000:]
        
        # Schedule cleanup
        for temp_path in temp_paths:
            background_tasks.add_task(cleanup_temp_file, temp_path)
        
        # Create summary
        summary = {
            "malicious_rate": malicious_count / len(files) if files else 0,
            "average_risk_score": sum(r['risk_score'] for r in detection_results) / len(detection_results) if detection_results else 0,
            "detection_types": {},
            "files_per_second": len(files) / total_processing_time if total_processing_time > 0 else 0
        }
        
        # Count detection types
        for result in detection_results:
            for detection_type in result.get('detection_types', []):
                summary["detection_types"][detection_type] = summary["detection_types"].get(detection_type, 0) + 1
        
        batch_result = BatchDetectionResult(
            total_files=len(files),
            malicious_files=malicious_count,
            clean_files=clean_count,
            processing_time=total_processing_time,
            results=results,
            summary=summary
        )
        
        logger.info(f"Batch detection completed: {len(files)} files, {malicious_count} malicious, {clean_count} clean")
        
        return batch_result
        
    except HTTPException:
        api_metrics["failed_detections"] += len(files)
        for temp_path in temp_paths:
            cleanup_temp_file(temp_path)
        raise
    except Exception as e:
        api_metrics["failed_detections"] += len(files)
        for temp_path in temp_paths:
            cleanup_temp_file(temp_path)
        logger.error(f"Batch detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )