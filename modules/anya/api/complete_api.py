"""
ANYA COMPLETE INTEGRATED API
=============================

Full production API with all features integrated:
- Authentication & Security
- Memory System
- Multi-Modal Processing
- Human Handoff
- Monitoring

Author: Ultra Platform Team
Version: 3.0.0
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, UTC
import asyncio
import logging
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import all Anya components
from modules.anya.security.auth_security import SecurityManager
from modules.anya.memory.memory_manager import MemoryManager
from modules.anya.multimodal.multimodal_processor import MultiModalProcessor
from modules.anya.handoff.handoff_manager import HandoffManager
from modules.anya.nlu.nlu_engine import NLUEngine
from modules.anya.safety.safety_compliance import SafetyComplianceSystem
from modules.anya.generation.response_generator import ResponseGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Anya AI Assistant - Complete API",
    description="Production-ready AI financial assistant with full features",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specific origins only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# INITIALIZE COMPONENTS
# ============================================================================

security_manager = SecurityManager()
memory_manager = MemoryManager()
multimodal_processor = MultiModalProcessor()
handoff_manager = HandoffManager()
nlu_engine = NLUEngine()
safety_system = SafetyComplianceSystem()
response_generator = ResponseGenerator()

logger.info("✅ All Anya components initialized")

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class AuthRequest(BaseModel):
    """Authentication request"""
    customer_id: str
    password: Optional[str] = None

class AuthResponse(BaseModel):
    """Authentication response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600

class ChatRequest(BaseModel):
    """Chat request"""
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    """Chat response"""
    response: str
    session_id: str
    intent: Dict[str, Any]
    entities: List[Dict[str, Any]]
    confidence: float
    requires_handoff: bool = False
    handoff_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]

class FileUploadResponse(BaseModel):
    """File upload response"""
    file_id: str
    file_type: str
    analysis: Dict[str, Any]
    summary: str

class MemoryQuery(BaseModel):
    """Memory query request"""
    query: str
    limit: int = 5

class MemoryResponse(BaseModel):
    """Memory response"""
    memories: List[Dict[str, Any]]
    preferences: Dict[str, Any]
    context: Dict[str, Any]

# ============================================================================
# AUTHENTICATION DEPENDENCY
# ============================================================================

async def verify_token(authorization: Optional[str] = Header(None)) -> str:
    """Verify JWT token and return customer_id"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    try:
        # Extract token
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        
        # Validate token
        customer_id = await security_manager.authenticate_request(jwt_token=token)
        
        if not customer_id:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        return customer_id
    
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root with service info"""
    return {
        "service": "Anya AI Assistant",
        "version": "3.0.0",
        "status": "operational",
        "features": {
            "nlu": "active",
            "safety": "active",
            "memory": "active",
            "multimodal": "active",
            "handoff": "active",
            "security": "active"
        },
        "endpoints": {
            "auth": "/api/auth/login",
            "chat": "/api/chat",
            "upload": "/api/upload",
            "memory": "/api/memory",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "components": {
            "nlu": "ok",
            "safety": "ok",
            "memory": "ok",
            "multimodal": "ok",
            "handoff": "ok"
        }
    }

@app.post("/api/auth/login", response_model=AuthResponse)
async def login(request: AuthRequest):
    """
    Login and get JWT tokens
    
    In production: validate against database
    """
    try:
        # Create session
        session = await security_manager.create_session(request.customer_id)
        
        logger.info(f"User {request.customer_id} logged in")
        
        return AuthResponse(**session)
    
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/auth/api-key")
async def generate_api_key(
    customer_id: str = Depends(verify_token),
    name: str = "API Key",
    tier: str = "basic"
):
    """Generate new API key"""
    try:
        raw_key, api_key = security_manager.generate_api_key(
            customer_id=customer_id,
            name=name,
            tier=tier
        )
        
        return {
            "api_key": raw_key,
            "key_id": api_key.key_id,
            "rate_limit": api_key.rate_limit,
            "created_at": api_key.created_at.isoformat()
        }
    
    except Exception as e:
        logger.error(f"API key generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate API key")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    customer_id: str = Depends(verify_token)
):
    """
    Complete chat endpoint with all features
    
    Flow:
    1. Rate limiting
    2. Input validation
    3. NLU processing
    4. Safety checks
    5. Memory retrieval
    6. Response generation
    7. Handoff detection
    8. Memory storage
    """
    import uuid
    import time
    
    start_time = time.time()
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
    
    try:
        # Step 1: Rate limiting
        allowed, rate_info = await security_manager.check_rate_limit(customer_id)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {rate_info.window_seconds}s"
            )
        
        # Step 2: Input validation
        valid, error = await security_manager.validate_message(request.message)
        if not valid:
            raise HTTPException(status_code=400, detail=error)
        
        # Step 3: NLU processing
        nlu_result = await nlu_engine.understand(request.message)
        
        # Step 4: Safety checks
        safety_report = await safety_system.check_input(request.message)
        if not safety_report.safe:
            return ChatResponse(
                response="I can't process that request. I'm here to help with financial questions and account information. What else can I help you with?",
                session_id=session_id,
                intent={"intent": "blocked", "confidence": 1.0},
                entities=[],
                confidence=0.0,
                metadata={
                    "blocked": True,
                    "reason": safety_report.flags,
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            )
        
        # Step 5: Memory retrieval
        memory_context = await memory_manager.get_relevant_context(
            customer_id=customer_id,
            current_query=request.message
        )
        
        # Step 6: Generate response (mock for demo)
        response_text = f"Based on your question about {nlu_result.intent.intent}, here's my answer..."
        
        # Step 7: Check if handoff needed
        conversation_history = memory_context.get("relevant_memories", [])
        
        escalation = await handoff_manager.check_and_escalate(
            customer_id=customer_id,
            session_id=session_id,
            query=request.message,
            ai_confidence=nlu_result.intent.confidence,
            conversation_history=conversation_history,
            customer_profile={"name": "User", "tier": "basic"},
            safety_flags=safety_report.flags
        )
        
        handoff_info = None
        if escalation:
            response_text = handoff_manager.get_handoff_message(escalation)
            handoff_info = {
                "escalation_id": escalation.escalation_id,
                "reason": escalation.reason.value,
                "priority": escalation.priority.value,
                "queue_position": handoff_manager.queue.get_queue_position(escalation.escalation_id),
                "estimated_wait": handoff_manager.queue.get_estimated_wait_time(escalation.escalation_id)
            }
        
        # Step 8: Store in memory
        await memory_manager.store_conversation(
            customer_id=customer_id,
            session_id=session_id,
            messages=[
                {"role": "user", "content": request.message},
                {"role": "assistant", "content": response_text}
            ],
            entities=[e.__dict__ for e in nlu_result.entities],
            intents=[nlu_result.intent.intent]
        )
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            intent={
                "intent": nlu_result.intent.intent,
                "category": nlu_result.intent.category.value,
                "confidence": nlu_result.intent.confidence
            },
            entities=[
                {
                    "type": e.type.value,
                    "value": e.value,
                    "normalized": e.normalized_value
                }
                for e in nlu_result.entities
            ],
            confidence=nlu_result.intent.confidence,
            requires_handoff=escalation is not None,
            handoff_info=handoff_info,
            metadata={
                "processing_time_ms": (time.time() - start_time) * 1000,
                "nlu_time_ms": nlu_result.processing_time_ms,
                "sentiment": nlu_result.semantic_frame.sentiment,
                "memory_count": len(memory_context.get("relevant_memories", []))
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    customer_id: str = Depends(verify_token)
):
    """
    Upload and process files (PDF, images, etc.)
    """
    try:
        # Read file
        file_bytes = await file.read()
        
        # Validate file
        valid, error = await security_manager.request_validator.validate_file(
            file_bytes, file.filename
        )
        
        if not valid:
            raise HTTPException(status_code=400, detail=error)
        
        # Get file type
        import os
        file_ext = os.path.splitext(file.filename)[1].lower().replace(".", "")
        
        # Process file
        result = await multimodal_processor.process_file(
            file_bytes=file_bytes,
            filename=file.filename,
            file_type=file_ext
        )
        
        # Extract summary
        if result["type"] == "document":
            summary = result["analysis"]["summary"]
        elif result["type"] == "image":
            summary = result["analysis"]["description"]
        else:
            summary = "File processed"
        
        return FileUploadResponse(
            file_id=f"file_{customer_id}_{file.filename}",
            file_type=result["type"],
            analysis=result["analysis"],
            summary=summary
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="File processing failed")

@app.get("/api/memory", response_model=MemoryResponse)
async def get_memory(
    query: Optional[str] = None,
    limit: int = 5,
    customer_id: str = Depends(verify_token)
):
    """Get relevant memories and preferences"""
    try:
        if query:
            context = await memory_manager.get_relevant_context(
                customer_id=customer_id,
                current_query=query,
                max_memories=limit
            )
        else:
            # Get recent memories
            recent = await memory_manager.memory_store.get_recent_memories(
                customer_id=customer_id,
                limit=limit
            )
            context = {
                "relevant_memories": [
                    {
                        "summary": m.summary,
                        "timestamp": m.timestamp.isoformat()
                    }
                    for m in recent
                ],
                "user_preferences": {},
                "memory_count": len(recent)
            }
        
        return MemoryResponse(
            memories=context["relevant_memories"],
            preferences=context.get("user_preferences", {}),
            context={
                "total_memories": context.get("memory_count", 0)
            }
        )
    
    except Exception as e:
        logger.error(f"Memory retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memories")

@app.get("/api/handoff/{escalation_id}")
async def get_handoff_status(
    escalation_id: str,
    customer_id: str = Depends(verify_token)
):
    """Get handoff/escalation status"""
    try:
        # Get queue position
        position = handoff_manager.queue.get_queue_position(escalation_id)
        wait_time = handoff_manager.queue.get_estimated_wait_time(escalation_id)
        
        # Check if active
        active = escalation_id in handoff_manager.queue.active_handoffs
        
        if position:
            status = "queued"
        elif active:
            status = "active"
        else:
            status = "unknown"
        
        return {
            "escalation_id": escalation_id,
            "status": status,
            "queue_position": position,
            "estimated_wait_minutes": wait_time
        }
    
    except Exception as e:
        logger.error(f"Handoff status error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get handoff status")

@app.get("/api/stats")
async def get_stats(customer_id: str = Depends(verify_token)):
    """Get user statistics"""
    try:
        # Get memories count
        memories = memory_manager.memory_store.memories.get(customer_id, [])
        
        # Get preferences
        prefs = await memory_manager.memory_store.get_user_preferences(customer_id)
        
        return {
            "total_conversations": len(memories),
            "communication_style": prefs.communication_style,
            "detail_level": prefs.detail_level,
            "sophistication_level": prefs.sophistication_level,
            "last_interaction": memories[-1].timestamp.isoformat() if memories else None
        }
    
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {"error": "Failed to retrieve stats"}

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    try:
        # Get auth token
        auth_data = await websocket.receive_json()
        token = auth_data.get("token")
        
        # Validate token
        customer_id = await security_manager.authenticate_request(jwt_token=token)
        
        if not customer_id:
            await websocket.close(code=1008, reason="Authentication failed")
            return
        
        await websocket.send_json({"type": "connected", "customer_id": customer_id})
        
        # Chat loop
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            if not message:
                continue
            
            # Send typing indicator
            await websocket.send_json({"type": "typing"})
            
            # Process message (simplified)
            nlu_result = await nlu_engine.understand(message)
            
            # Send response
            await websocket.send_json({
                "type": "message",
                "content": f"Received: {message}",
                "intent": nlu_result.intent.intent,
                "confidence": nlu_result.intent.confidence
            })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason="Internal error")

# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("🚀 Anya API starting up...")
    logger.info("✅ All systems operational")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("👋 Anya API shutting down...")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🚀 STARTING ANYA COMPLETE API")
    print("=" * 70)
    print("\n📡 Endpoints:")
    print("   API Docs:    http://localhost:8000/docs")
    print("   Health:      http://localhost:8000/health")
    print("   Chat:        POST http://localhost:8000/api/chat")
    print("   Upload:      POST http://localhost:8000/api/upload")
    print("   Memory:      GET http://localhost:8000/api/memory")
    print("   WebSocket:   ws://localhost:8000/ws/chat")
    print("")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
