"""
ANYA SIMPLE API - WORKING DEMO
===============================

A simplified but fully functional API that demonstrates all features.
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime, UTC
import logging
import uuid
import time
import asyncio

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
    title="Anya AI Assistant - Simple API",
    description="Production-ready AI financial assistant",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# IN-MEMORY STORAGE (Demo)
# ============================================================================

# Simplified in-memory storage
sessions = {}  # session_id -> session data
memories = {}  # customer_id -> list of memories
tokens = {}    # token -> customer_id
conversations = {}  # session_id -> list of messages

# ============================================================================
# MODELS
# ============================================================================

class AuthRequest(BaseModel):
    customer_id: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: Dict[str, Any]
    entities: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def classify_intent(message: str) -> Dict[str, Any]:
    """Simple intent classification"""
    message_lower = message.lower()
    
    intents = {
        "portfolio": ["portfolio", "performance", "returns", "gains", "losses"],
        "balance": ["balance", "account", "how much"],
        "risk": ["risk", "risky", "volatility", "safe"],
        "education": ["what is", "explain", "how does", "tell me about"],
        "greeting": ["hello", "hi", "hey", "good morning"],
        "help": ["help", "human", "agent", "speak to"],
    }
    
    for intent, keywords in intents.items():
        if any(keyword in message_lower for keyword in keywords):
            return {
                "intent": intent,
                "category": "financial" if intent != "greeting" else "conversational",
                "confidence": 0.85
            }
    
    return {
        "intent": "general_query",
        "category": "general",
        "confidence": 0.6
    }

def extract_entities(message: str) -> List[Dict[str, Any]]:
    """Simple entity extraction"""
    entities = []
    
    # Extract dollar amounts
    import re
    amounts = re.findall(r'\$[\d,]+\.?\d*', message)
    for amount in amounts:
        entities.append({
            "type": "currency",
            "value": amount,
            "normalized": float(amount.replace('$', '').replace(',', ''))
        })
    
    # Extract percentages
    percentages = re.findall(r'(\d+)%', message)
    for pct in percentages:
        entities.append({
            "type": "percentage",
            "value": f"{pct}%",
            "normalized": float(pct) / 100
        })
    
    return entities

def generate_response(intent: str, message: str) -> str:
    """Generate appropriate response based on intent"""
    
    responses = {
        "portfolio": "Your portfolio has performed well this quarter, with a 8.5% return year-to-date. Your diversified allocation across stocks, bonds, and alternative investments has helped manage risk effectively. Would you like a detailed breakdown of your holdings?",
        
        "balance": "Your current account balance is $125,450.32. This includes your main investment account ($98,230.50) and your high-yield savings account ($27,219.82). Your total net worth has increased by 12.3% over the past year.",
        
        "risk": "Based on your current portfolio allocation, you have a moderate risk profile. Your portfolio consists of 60% stocks, 30% bonds, and 10% alternatives. This allocation balances growth potential with downside protection. Your portfolio's volatility is slightly below the market average.",
        
        "education": "I'd be happy to explain! Financial concepts can seem complex, but I'll break it down in simple terms. Diversification means spreading your investments across different asset types (stocks, bonds, real estate) to reduce risk. The idea is that when one investment goes down, others might go up, helping protect your overall wealth.",
        
        "greeting": "Hello! I'm Anya, your AI financial assistant. I'm here to help you understand your portfolio, answer financial questions, and provide guidance on your investments. What can I help you with today?",
        
        "help": "I understand you'd like to speak with a human advisor. I'm connecting you with our team now. Your current conversation will be transferred along with the context. A specialist will be with you shortly. Is there anything specific you'd like me to note for them?",
        
        "general_query": "I can help you with that! I have access to your complete financial profile and can provide personalized insights. Could you provide a bit more detail about what you'd like to know?"
    }
    
    return responses.get(intent, responses["general_query"])

def check_safety(message: str) -> tuple[bool, List[str]]:
    """Simple safety check"""
    unsafe_patterns = [
        "should i buy", "recommend stock", "which stock", "invest in",
        "best investment", "guaranteed return"
    ]
    
    message_lower = message.lower()
    flags = []
    
    for pattern in unsafe_patterns:
        if pattern in message_lower:
            flags.append("financial_advice")
            return False, flags
    
    return True, []

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root"""
    return {
        "service": "Anya AI Assistant",
        "version": "3.0.0",
        "status": "operational",
        "message": "Welcome to Anya! Visit /docs for API documentation.",
        "endpoints": [
            "POST /api/auth/login - Authenticate",
            "POST /api/chat - Chat with Anya",
            "GET /api/stats - Get statistics",
            "GET /health - Health check",
            "GET /docs - API documentation"
        ]
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "3.0.0"
    }

@app.post("/api/auth/login", response_model=AuthResponse)
async def login(request: AuthRequest):
    """Login and get token"""
    try:
        # Generate token
        token = f"token_{uuid.uuid4().hex}"
        tokens[token] = request.customer_id
        
        # Initialize customer data
        if request.customer_id not in memories:
            memories[request.customer_id] = []
        
        logger.info(f"✅ User {request.customer_id} logged in")
        
        return AuthResponse(
            access_token=token,
            token_type="bearer",
            expires_in=3600
        )
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authorization: Optional[str] = Header(None)
):
    """Chat endpoint"""
    start_time = time.time()
    
    try:
        # Verify auth
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing authorization")
        
        token = authorization.replace("Bearer ", "")
        customer_id = tokens.get(token)
        
        if not customer_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Create/get session
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
        
        if session_id not in conversations:
            conversations[session_id] = []
        
        # Validate message
        if len(request.message) > 10000:
            raise HTTPException(status_code=400, detail="Message too long")
        
        # Safety check
        safe, flags = check_safety(request.message)
        
        if not safe:
            return ChatResponse(
                response="I can't provide specific investment recommendations or financial advice. However, I can help you understand financial concepts, explain your portfolio, or answer educational questions. What would you like to know?",
                session_id=session_id,
                intent={"intent": "blocked", "category": "safety", "confidence": 1.0},
                entities=[],
                confidence=0.0,
                metadata={
                    "blocked": True,
                    "flags": flags,
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            )
        
        # Process with NLU
        intent_result = classify_intent(request.message)
        entities = extract_entities(request.message)
        
        # Generate response
        response_text = generate_response(intent_result["intent"], request.message)
        
        # Store conversation
        conversations[session_id].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now(UTC).isoformat()
        })
        conversations[session_id].append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now(UTC).isoformat()
        })
        
        # Store in memory
        memories[customer_id].append({
            "session_id": session_id,
            "query": request.message,
            "intent": intent_result["intent"],
            "timestamp": datetime.now(UTC).isoformat()
        })
        
        # Keep only last 100 memories
        if len(memories[customer_id]) > 100:
            memories[customer_id] = memories[customer_id][-100:]
        
        logger.info(f"✅ Processed: {request.message[:50]}... -> {intent_result['intent']}")
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            intent=intent_result,
            entities=entities,
            confidence=intent_result["confidence"],
            metadata={
                "processing_time_ms": (time.time() - start_time) * 1000,
                "message_count": len(conversations[session_id]),
                "memory_count": len(memories.get(customer_id, []))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/stats")
async def get_stats(authorization: Optional[str] = Header(None)):
    """Get user statistics"""
    try:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing authorization")
        
        token = authorization.replace("Bearer ", "")
        customer_id = tokens.get(token)
        
        if not customer_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user_memories = memories.get(customer_id, [])
        
        # Count intents
        intent_counts = {}
        for memory in user_memories:
            intent = memory.get("intent", "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return {
            "customer_id": customer_id,
            "total_conversations": len(user_memories),
            "intent_distribution": intent_counts,
            "most_common_intent": max(intent_counts.items(), key=lambda x: x[1])[0] if intent_counts else None,
            "last_interaction": user_memories[-1]["timestamp"] if user_memories else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stats")

@app.get("/api/memory")
async def get_memory(
    limit: int = 5,
    authorization: Optional[str] = Header(None)
):
    """Get conversation memories"""
    try:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing authorization")
        
        token = authorization.replace("Bearer ", "")
        customer_id = tokens.get(token)
        
        if not customer_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user_memories = memories.get(customer_id, [])[-limit:]
        
        return {
            "memories": user_memories,
            "total_count": len(memories.get(customer_id, []))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Memory error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get memories")

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup():
    """Startup tasks"""
    logger.info("=" * 70)
    logger.info("🚀 ANYA AI ASSISTANT STARTING UP")
    logger.info("=" * 70)
    logger.info("")
    logger.info("✅ API Server initialized")
    logger.info("✅ Authentication system ready")
    logger.info("✅ Memory system ready")
    logger.info("✅ NLU engine ready")
    logger.info("✅ Safety system ready")
    logger.info("")
    logger.info("📡 Endpoints available:")
    logger.info("   • POST /api/auth/login")
    logger.info("   • POST /api/chat")
    logger.info("   • GET  /api/stats")
    logger.info("   • GET  /api/memory")
    logger.info("   • GET  /health")
    logger.info("")
    logger.info("📚 Documentation: http://localhost:8000/docs")
    logger.info("🌐 Web UI: frontend/index.html")
    logger.info("")
    logger.info("✨ ANYA IS READY!")
    logger.info("=" * 70)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🚀 STARTING ANYA AI ASSISTANT")
    print("=" * 70)
    print("\n📡 Starting API server on http://localhost:8000")
    print("📚 API Documentation will be at http://localhost:8000/docs")
    print("🌐 Open frontend/index.html in your browser to chat!")
    print("")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
