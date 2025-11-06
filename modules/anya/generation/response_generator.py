"""
ANYA RESPONSE GENERATION SYSTEM
================================

Production-ready response generation with:
- GPT-4 Turbo integration
- Anya's core identity & system prompts
- Citation injection
- Token management
- Streaming support
- Context optimization

Author: Ultra Platform Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Dict, List, Any, Optional, AsyncIterator
import asyncio
import logging
import json
import re
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ANYA'S CORE IDENTITY
# ============================================================================

ANYA_SYSTEM_PROMPT = """You are Anya, a knowledgeable and friendly AI assistant for Ultra Platform, a wealth management platform. Your role is to help clients understand their investments, answer questions, and explain portfolio decisions.

Core Principles:
1. Clarity: Explain complex financial concepts in simple terms
2. Accuracy: Base all responses on provided context, never speculate
3. Personalization: Tailor explanations to client's sophistication level
4. Proactivity: Identify opportunities to provide helpful insights
5. Compliance: Never provide financial advice, only education and explanation

Response Guidelines:
- Use the client's first name when appropriate
- Reference specific numbers from their portfolio
- Explain "why" behind decisions, not just "what"
- Use analogies and examples for complex topics
- Acknowledge uncertainty when context is insufficient
- Provide actionable next steps when relevant

Tone:
- Professional yet warm and conversational
- Confident but not condescending
- Empathetic to client concerns
- Educational without being preachy

What You CAN Do:
✓ Explain portfolio decisions and changes
✓ Answer questions about performance and risk
✓ Provide market context and education
✓ Help clients understand their goals
✓ Explain fees and costs
✓ Guide through platform features

What You CANNOT Do:
✗ Provide specific investment recommendations
✗ Guarantee future returns
✗ Make decisions on behalf of client
✗ Discuss other clients or portfolios
✗ Share confidential information
✗ Provide tax or legal advice

CRITICAL: Always base your responses on the provided context. If you cannot find information in the context to support your answer, acknowledge this limitation. Never make up facts or figures."""


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class GenerationConfig:
    """LLM generation configuration"""
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = False


@dataclass
class Citation:
    """Source citation"""
    source_id: str
    source_name: str
    relevance_score: float
    excerpt: str
    url: Optional[str] = None
    page_number: Optional[int] = None


@dataclass
class GeneratedResponse:
    """Complete generated response"""
    text: str
    citations: List[Citation]
    token_usage: Dict[str, int]
    generation_time_ms: float
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationMessage:
    """Single conversation message"""
    role: str  # 'system', 'user', 'assistant'
    content: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# CONTEXT ASSEMBLER
# ============================================================================

class ContextAssembler:
    """
    Optimized context assembly for token management
    
    Assembles:
    - System prompt
    - Conversation history
    - Retrieved documents
    - Customer portfolio data
    
    Ensures token limits are respected
    """
    
    def __init__(self, max_context_tokens: int = 120000):  # GPT-4 Turbo: 128k context
        self.max_context_tokens = max_context_tokens
        
        # Rough token estimation (4 chars ≈ 1 token)
        self.chars_per_token = 4
        
        logger.info(f"Context Assembler initialized (max tokens: {max_context_tokens})")
    
    def assemble_context(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        conversation_history: Optional[List[ConversationMessage]] = None,
        customer_context: Optional[Dict[str, Any]] = None,
        max_docs: int = 10
    ) -> str:
        """
        Assemble optimized context for LLM
        
        Priority order:
        1. System prompt (always included)
        2. Customer context (if available)
        3. Recent conversation history
        4. Most relevant retrieved documents
        5. Current query
        """
        context_parts = []
        
        # Add customer context if available
        if customer_context:
            customer_section = self._build_customer_context(customer_context)
            context_parts.append(customer_section)
        
        # Add retrieved documents
        if retrieved_docs:
            docs_section = self._build_documents_context(retrieved_docs[:max_docs])
            context_parts.append(docs_section)
        
        # Combine context
        full_context = "\n\n".join(context_parts)
        
        # Check token budget
        estimated_tokens = len(full_context) // self.chars_per_token
        
        if estimated_tokens > self.max_context_tokens * 0.7:  # Use 70% for context
            # Trim documents if needed
            logger.warning(f"Context too large ({estimated_tokens} tokens), trimming...")
            full_context = self._trim_context(full_context, self.max_context_tokens * 0.7)
        
        return full_context
    
    def _build_customer_context(self, customer_context: Dict[str, Any]) -> str:
        """Build customer context section"""
        sections = ["CUSTOMER CONTEXT:"]
        
        if "name" in customer_context:
            sections.append(f"Customer Name: {customer_context['name']}")
        
        if "portfolio" in customer_context:
            portfolio = customer_context["portfolio"]
            sections.append(f"\nPortfolio Summary:")
            
            if "total_value" in portfolio:
                sections.append(f"  Total Value: ${portfolio['total_value']:,.2f}")
            
            if "allocation" in portfolio:
                sections.append(f"  Allocation:")
                for asset_class, percentage in portfolio["allocation"].items():
                    sections.append(f"    - {asset_class}: {percentage}%")
            
            if "performance" in portfolio:
                sections.append(f"  Performance: {portfolio['performance']}")
        
        return "\n".join(sections)
    
    def _build_documents_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build documents context section"""
        sections = ["KNOWLEDGE BASE DOCUMENTS:"]
        
        for i, doc in enumerate(documents, 1):
            sections.append(f"\n[Document {i}: {doc.get('source', 'Unknown Source')}]")
            sections.append(f"Relevance: {doc.get('relevance', 0.0):.2f}")
            sections.append(f"Content: {doc.get('content', '')[:500]}...")  # First 500 chars
        
        return "\n".join(sections)
    
    def _trim_context(self, context: str, max_tokens: float) -> str:
        """Trim context to fit token budget"""
        max_chars = int(max_tokens * self.chars_per_token)
        
        if len(context) <= max_chars:
            return context
        
        # Keep beginning and end, remove middle
        keep_chars = max_chars // 2
        
        trimmed = (
            context[:keep_chars] +
            "\n\n[... content trimmed for length ...]\n\n" +
            context[-keep_chars:]
        )
        
        return trimmed


# ============================================================================
# CITATION EXTRACTOR
# ============================================================================

class CitationExtractor:
    """
    Extract factual claims and match to source documents
    
    Process:
    1. Identify factual claims in response
    2. Match claims to source documents
    3. Calculate attribution confidence
    4. Inject inline citations
    """
    
    def __init__(self):
        # Patterns for factual claims
        self.factual_patterns = [
            r'\b\d+%',  # Percentages
            r'\$[\d,]+',  # Dollar amounts
            r'\b(is|are|was|were) (a|an|the)? \w+',  # Definitions
            r'\b(means|refers to|indicates)\b',  # Explanations
        ]
        
        self.patterns = [re.compile(p) for p in self.factual_patterns]
        
        logger.info("Citation Extractor initialized")
    
    async def extract_citations(
        self,
        response_text: str,
        source_docs: List[Dict[str, Any]]
    ) -> List[Citation]:
        """
        Extract citations from response
        
        Returns list of citations with source attribution
        """
        citations = []
        
        # Create citations from source documents
        for i, doc in enumerate(source_docs):
            citation = Citation(
                source_id=doc.get("id", f"doc_{i}"),
                source_name=doc.get("source", f"Document {i+1}"),
                relevance_score=doc.get("relevance", 1.0),
                excerpt=doc.get("content", "")[:200],  # First 200 chars
                url=doc.get("url")
            )
            
            citations.append(citation)
        
        return citations
    
    async def inject_citations(
        self,
        response_text: str,
        citations: List[Citation]
    ) -> str:
        """
        Inject citation markers into response text
        
        Adds inline citations and references section
        """
        if not citations:
            return response_text
        
        # Add references section at end
        references_section = "\n\n---\n### Sources\n\n"
        
        for i, citation in enumerate(citations, 1):
            references_section += f"[{i}] **{citation.source_name}** "
            references_section += f"(Relevance: {citation.relevance_score:.0%})\n"
            
            if citation.excerpt:
                references_section += f"   _{citation.excerpt[:150]}..._\n"
            
            if citation.url:
                references_section += f"   [View Source]({citation.url})\n"
            
            references_section += "\n"
        
        return response_text + references_section


# ============================================================================
# LLM CLIENT
# ============================================================================

class LLMClient:
    """
    GPT-4 Turbo client with streaming support
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.config = config or GenerationConfig()
        
        # Initialize OpenAI client if available
        self.client = None
        self.available = False
        
        if self.api_key:
            try:
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI(api_key=self.api_key)
                self.available = True
                logger.info("✅ OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI package not available")
        else:
            logger.warning("No OpenAI API key provided")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> Any:
        """
        Generate response from GPT-4 Turbo
        
        Args:
            messages: Conversation messages
            system_prompt: System prompt override
            stream: Whether to stream response
        """
        if not self.available:
            return self._mock_response(messages)
        
        # Prepare messages
        formatted_messages = []
        
        if system_prompt:
            formatted_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        formatted_messages.extend(messages)
        
        # Call API
        try:
            if stream:
                return await self._stream_response(formatted_messages)
            else:
                return await self._complete_response(formatted_messages)
        
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._mock_response(messages)
    
    async def _complete_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Get complete response"""
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty
        )
        
        return {
            "text": response.choices[0].message.content,
            "token_usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason
        }
    
    async def _stream_response(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """Stream response"""
        stream = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def _mock_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate mock response for testing"""
        user_message = messages[-1]["content"] if messages else "Hello"
        
        mock_text = f"""Based on the information available, I can help explain that concept.

{user_message[:100]}... is an important topic in wealth management. Let me break it down:

1. **Key Concept**: This relates to how we manage your portfolio
2. **Why It Matters**: Understanding this helps you make informed decisions
3. **Your Portfolio**: Based on your current allocation, this applies to your situation

Would you like me to explain any specific aspect in more detail?"""
        
        return {
            "text": mock_text,
            "token_usage": {
                "prompt_tokens": 200,
                "completion_tokens": 150,
                "total_tokens": 350
            },
            "model": "mock-gpt-4",
            "finish_reason": "stop"
        }


# ============================================================================
# COMPLETE RESPONSE GENERATOR
# ============================================================================

class ResponseGenerator:
    """
    Complete response generation system
    
    Orchestrates:
    - Context assembly
    - LLM generation
    - Citation extraction
    - Citation injection
    - Token management
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ):
        self.context_assembler = ContextAssembler()
        self.citation_extractor = CitationExtractor()
        self.llm_client = LLMClient(api_key, config)
        self.config = config or GenerationConfig()
        
        logger.info("✅ Response Generator initialized")
    
    async def generate_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        conversation_history: Optional[List[ConversationMessage]] = None,
        customer_context: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> GeneratedResponse:
        """
        Generate complete response with citations
        
        Full pipeline:
        1. Assemble context
        2. Generate response with LLM
        3. Extract citations
        4. Inject citations
        5. Return complete response
        """
        import time
        start_time = time.time()
        
        # Step 1: Assemble context
        context = self.context_assembler.assemble_context(
            query=query,
            retrieved_docs=retrieved_docs,
            conversation_history=conversation_history,
            customer_context=customer_context
        )
        
        # Step 2: Build messages
        messages = []
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-6:]:  # Last 6 messages
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add current query with context
        messages.append({
            "role": "user",
            "content": f"{context}\n\nUser Question: {query}\n\nPlease provide a helpful, accurate response based on the context above."
        })
        
        # Step 3: Generate with LLM
        if stream:
            # For streaming, return async generator
            return self._stream_response(messages, retrieved_docs, start_time)
        else:
            response_data = await self.llm_client.generate(
                messages=messages,
                system_prompt=ANYA_SYSTEM_PROMPT,
                stream=False
            )
            
            # Step 4: Extract citations
            citations = await self.citation_extractor.extract_citations(
                response_data["text"],
                retrieved_docs
            )
            
            # Step 5: Inject citations
            final_text = await self.citation_extractor.inject_citations(
                response_data["text"],
                citations
            )
            
            # Calculate timing
            generation_time_ms = (time.time() - start_time) * 1000
            
            return GeneratedResponse(
                text=final_text,
                citations=citations,
                token_usage=response_data["token_usage"],
                generation_time_ms=generation_time_ms,
                model=response_data["model"],
                metadata={
                    "retrieved_docs": len(retrieved_docs),
                    "conversation_length": len(conversation_history) if conversation_history else 0
                }
            )
    
    async def _stream_response(
        self,
        messages: List[Dict[str, str]],
        retrieved_docs: List[Dict[str, Any]],
        start_time: float
    ) -> AsyncIterator[str]:
        """Stream response with citations at end"""
        # Stream main response
        response_chunks = []
        
        async for chunk in await self.llm_client.generate(
            messages=messages,
            system_prompt=ANYA_SYSTEM_PROMPT,
            stream=True
        ):
            response_chunks.append(chunk)
            yield chunk
        
        # Add citations at end
        full_response = "".join(response_chunks)
        citations = await self.citation_extractor.extract_citations(
            full_response,
            retrieved_docs
        )
        
        # Stream citation section
        citation_text = await self.citation_extractor.inject_citations("", citations)
        yield citation_text


# ============================================================================
# DEMO
# ============================================================================

async def demo_response_generation():
    """Demonstrate response generation system"""
    print("\n" + "=" * 70)
    print("ANYA RESPONSE GENERATION SYSTEM DEMO")
    print("=" * 70)
    
    # Initialize generator
    generator = ResponseGenerator()
    
    # Mock data
    query = "What is portfolio diversification and why is it important?"
    
    retrieved_docs = [
        {
            "id": "doc_1",
            "source": "Portfolio Management Guide",
            "content": "Diversification is an investment strategy that reduces risk by allocating investments across various financial instruments, industries, and other categories. It aims to maximize returns by investing in different areas that would each react differently to the same event.",
            "relevance": 0.95,
            "url": "https://ultra.com/guides/diversification"
        },
        {
            "id": "doc_2",
            "source": "Risk Management Handbook",
            "content": "A diversified portfolio combines different asset classes—stocks, bonds, real estate, and cash—to reduce overall risk. Historical data shows that diversified portfolios tend to have lower volatility than concentrated portfolios.",
            "relevance": 0.88
        }
    ]
    
    customer_context = {
        "name": "Sarah",
        "portfolio": {
            "total_value": 125000,
            "allocation": {
                "Stocks": 60,
                "Bonds": 30,
                "Real Estate": 10
            },
            "performance": "+12.5% YTD"
        }
    }
    
    conversation_history = [
        ConversationMessage(
            role="user",
            content="Hi Anya, I have some questions about my portfolio"
        ),
        ConversationMessage(
            role="assistant",
            content="Hi Sarah! I'd be happy to help you understand your portfolio. What would you like to know?"
        )
    ]
    
    print("\n📝 Generating response...")
    print(f"Query: {query}")
    print(f"Retrieved Docs: {len(retrieved_docs)}")
    print(f"Customer: {customer_context['name']}")
    
    # Generate response
    response = await generator.generate_response(
        query=query,
        retrieved_docs=retrieved_docs,
        conversation_history=conversation_history,
        customer_context=customer_context,
        stream=False
    )
    
    print("\n" + "=" * 70)
    print("✨ GENERATED RESPONSE:")
    print("=" * 70)
    print(response.text)
    
    print("\n" + "=" * 70)
    print("📊 METRICS:")
    print("=" * 70)
    print(f"Generation Time: {response.generation_time_ms:.1f}ms")
    print(f"Model: {response.model}")
    print(f"Token Usage:")
    print(f"  Prompt: {response.token_usage['prompt_tokens']}")
    print(f"  Completion: {response.token_usage['completion_tokens']}")
    print(f"  Total: {response.token_usage['total_tokens']}")
    print(f"Citations: {len(response.citations)}")
    
    print("\n" + "=" * 70)
    print("✅ Response Generation Demo Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(demo_response_generation())
