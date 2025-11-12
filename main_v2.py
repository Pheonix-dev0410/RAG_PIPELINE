"""
Enhanced FastAPI Application with Dual-Mode RAG
Fixes: Shared memory bug, proper session management, rate limiting
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime
import os
import logging
import threading
import time
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.rag_pipeline_v2 import ingest_pdf, query_rag_system, check_rag_health
from app.session_manager import get_session_manager, start_cleanup_thread

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_pipeline_v2.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self._lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Clean old requests
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > window_start
            ]
            
            # Check limit
            if len(self.requests[client_id]) >= self.max_requests:
                return False
            
            # Add new request
            self.requests[client_id].append(now)
            return True
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests in window"""
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            recent_requests = [
                req_time for req_time in self.requests[client_id]
                if req_time > window_start
            ]
            return max(0, self.max_requests - len(recent_requests))

rate_limiter = RateLimiter(max_requests=60, window_seconds=60)

# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

PDF_PATH = "temp.pdf"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    try:
        logger.info("=" * 80)
        logger.info("🚀 Starting Enhanced RAG Pipeline Application v2.0")
        logger.info(f"Startup time: {datetime.now()}")
        logger.info("=" * 80)
        
        # Start session cleanup thread
        start_cleanup_thread(interval_seconds=300)
        
        # Load PDF if exists
        if os.path.exists(PDF_PATH):
            try:
                logger.info(f"Loading existing PDF: {PDF_PATH}")
                ingest_pdf(PDF_PATH, backup=False)
                logger.info("✅ PDF loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load PDF: {e}")
        else:
            logger.warning(f"No PDF found at {PDF_PATH}. Upload one to start.")
        
        logger.info("✅ Application startup complete")
        
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("=" * 80)
    logger.info("🛑 Shutting down RAG Pipeline Application")
    logger.info(f"Shutdown time: {datetime.now()}")
    
    # Print session stats
    stats = get_session_manager().get_stats()
    logger.info(f"Session stats: {stats}")
    logger.info("=" * 80)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    lifespan=lifespan,
    title="Enhanced RAG Pipeline API v2",
    description="Dual-mode RAG system for video recommendations and transcript-based Q&A",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_name: Optional[str] = Field(None, description="User's name for personalization")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "I want to learn fractions",
                "session_id": None,
                "user_name": "Pranav"
            }
        }

class QueryResponse(BaseModel):
    answer: Optional[str] = Field(None, description="Text answer for transcript Q&A")
    video_link: Optional[str] = Field(None, description="Video link for new learning requests")
    video_title: Optional[str] = Field(None, description="Video title if link provided")
    session_id: str = Field(..., description="Session ID for future requests")
    mode: str = Field(..., description="Mode used: 'video' or 'transcript'")
    status: str = Field(default="success")

class UploadResponse(BaseModel):
    message: str
    file_size: int
    chunks_indexed: int
    status: str = "success"

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    faiss_index_loaded: bool
    active_sessions: int
    pdf_exists: bool

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    try:
        session_stats = get_session_manager().get_stats()
        return {
            "message": "Enhanced RAG Pipeline API v2.0",
            "docs_url": "/docs",
            "features": [
                "Dual-mode operation (video search + transcript Q&A)",
                "Per-user session management",
                "Natural conversational responses",
                "User name personalization",
                "Rate limiting protection"
            ],
            "active_sessions": session_stats["active_sessions"],
            "pdf_loaded": os.path.exists(os.path.join("faiss_index", "index.faiss")),
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF containing video metadata and transcripts.
    
    The PDF should contain:
    - Video titles, links, and descriptions (for video search)
    - Video transcripts (for follow-up Q&A)
    """
    try:
        logger.info(f"Upload request: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")
        
        # Read file
        content = await file.read()
        file_size = len(content)
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        if file_size > 100 * 1024 * 1024:  # 100MB
            raise HTTPException(status_code=400, detail="File too large (max 100MB)")
        
        # Write to disk
        logger.info(f"Writing file: {file_size} bytes")
        with open(PDF_PATH, "wb") as f:
            f.write(content)
        
        # Ingest PDF
        logger.info("Ingesting PDF...")
        ingest_pdf(PDF_PATH, backup=True)
        
        # Count chunks (approximate)
        from app.pdf_loader_v2 import load_and_split_pdf
        docs = load_and_split_pdf(PDF_PATH)
        
        logger.info(f"✅ Upload complete: {len(docs)} chunks")
        
        return UploadResponse(
            message=f"Successfully uploaded and indexed '{file.filename}'",
            file_size=file_size,
            chunks_indexed=len(docs),
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    x_forwarded_for: Optional[str] = Header(None),
    client_host: Optional[str] = Header(None)
):
    """
    Ask a question to the RAG system.
    
    The system automatically detects:
    - New learning requests → Returns video link
    - Follow-up questions → Returns contextual answer from transcript
    """
    try:
        # Rate limiting
        client_id = x_forwarded_for or client_host or "unknown"
        if not rate_limiter.is_allowed(client_id):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        logger.info(f"Query: '{request.question}' (session: {request.session_id})")
        
        # Validate
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Check if RAG system is ready
        health = check_rag_health()
        if not health.get("faiss_index_exists"):
            raise HTTPException(
                status_code=503,
                detail="No PDF loaded. Please upload a PDF first."
            )
        
        # Query RAG system
        result = query_rag_system(
            question=request.question,
            session_id=request.session_id,
            user_name=request.user_name
        )
        
        logger.info(
            f"✅ Response generated: mode={result['mode']}, "
            f"session={result['session_id']}"
        )
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        health = check_rag_health()
        session_stats = health.get("session_manager_stats", {})
        
        return HealthResponse(
            status=health.get("status", "unknown"),
            timestamp=datetime.now().isoformat(),
            faiss_index_loaded=health.get("faiss_index_exists", False),
            active_sessions=session_stats.get("active_sessions", 0),
            pdf_exists=os.path.exists(PDF_PATH)
        )
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/sessions/stats")
async def session_stats():
    """Get session manager statistics"""
    try:
        stats = get_session_manager().get_stats()
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            **stats
        }
    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get stats")


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    try:
        manager = get_session_manager()
        manager.delete_session(session_id)
        return {
            "status": "success",
            "message": f"Session {session_id} deleted"
        }
    except Exception as e:
        logger.error(f"Delete session error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete session")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

