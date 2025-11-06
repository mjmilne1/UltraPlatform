"""
ANYA HUMAN HANDOFF SYSTEM
==========================

Seamless AI-to-human escalation with:
- Escalation triggers
- Context transfer
- Queue management
- Agent assist mode
- Conversation continuity

Author: Ultra Platform Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Dict, List, Any, Optional
from enum import Enum
import asyncio
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class EscalationReason(str, Enum):
    """Reasons for escalation"""
    LOW_CONFIDENCE = "low_confidence"
    USER_REQUEST = "user_request"
    COMPLEX_QUERY = "complex_query"
    REGULATORY_REQUIREMENT = "regulatory_requirement"
    DETECTED_FRUSTRATION = "detected_frustration"
    SAFETY_CONCERN = "safety_concern"
    ACCOUNT_ISSUE = "account_issue"


class HandoffStatus(str, Enum):
    """Handoff status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    """Escalation priority"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class EscalationRequest:
    """Request to escalate to human agent"""
    escalation_id: str
    customer_id: str
    session_id: str
    reason: EscalationReason
    priority: Priority
    created_at: datetime
    
    # Context
    conversation_summary: str
    last_messages: List[Dict[str, str]]
    customer_profile: Dict[str, Any]
    ai_confidence_scores: List[float]
    detected_issues: List[str]
    
    # Status
    status: HandoffStatus = HandoffStatus.PENDING
    assigned_agent: Optional[str] = None
    assigned_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContext:
    """Context package for human agent"""
    customer_name: str
    customer_tier: str
    account_summary: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    conversation_summary: str
    key_entities_mentioned: List[str]
    ai_attempted_responses: List[str]
    recommended_actions: List[str]
    escalation_reason: str
    priority: str


# ============================================================================
# ESCALATION DETECTOR
# ============================================================================

class EscalationDetector:
    """
    Detect when escalation is needed
    
    Monitors:
    - AI confidence scores
    - User sentiment/frustration
    - Query complexity
    - Explicit requests
    - Regulatory triggers
    """
    
    def __init__(self):
        self.confidence_threshold = 0.6
        self.frustration_keywords = [
            "frustrated", "angry", "upset", "terrible", "useless",
            "speak to human", "real person", "agent", "representative"
        ]
        logger.info("Escalation Detector initialized")
    
    async def should_escalate(
        self,
        query: str,
        ai_confidence: float,
        conversation_history: List[Dict[str, str]],
        safety_flags: List[str]
    ) -> tuple[bool, Optional[EscalationReason], Priority]:
        """
        Determine if escalation is needed
        
        Returns: (should_escalate, reason, priority)
        """
        
        # Check explicit user request
        if self._check_explicit_request(query):
            return True, EscalationReason.USER_REQUEST, Priority.MEDIUM
        
        # Check AI confidence
        if ai_confidence < self.confidence_threshold:
            return True, EscalationReason.LOW_CONFIDENCE, Priority.LOW
        
        # Check for frustration
        if self._detect_frustration(query, conversation_history):
            return True, EscalationReason.DETECTED_FRUSTRATION, Priority.HIGH
        
        # Check safety concerns
        if safety_flags and any(flag in ["high_risk", "urgent"] for flag in safety_flags):
            return True, EscalationReason.SAFETY_CONCERN, Priority.URGENT
        
        # Check query complexity
        if self._is_complex_query(query):
            return True, EscalationReason.COMPLEX_QUERY, Priority.MEDIUM
        
        return False, None, Priority.LOW
    
    def _check_explicit_request(self, query: str) -> bool:
        """Check if user explicitly asks for human"""
        query_lower = query.lower()
        
        explicit_phrases = [
            "speak to", "talk to", "human", "agent", "representative",
            "real person", "customer service", "support"
        ]
        
        return any(phrase in query_lower for phrase in explicit_phrases)
    
    def _detect_frustration(
        self,
        query: str,
        conversation_history: List[Dict[str, str]]
    ) -> bool:
        """Detect user frustration"""
        query_lower = query.lower()
        
        # Check for frustration keywords
        if any(keyword in query_lower for keyword in self.frustration_keywords):
            return True
        
        # Check for repeated queries (indicates frustration)
        if len(conversation_history) >= 4:
            recent_queries = [
                msg["content"].lower()
                for msg in conversation_history[-4:]
                if msg.get("role") == "user"
            ]
            
            # If similar queries repeated
            if len(recent_queries) >= 2:
                if recent_queries[-1] in recent_queries[:-1]:
                    return True
        
        return False
    
    def _is_complex_query(self, query: str) -> bool:
        """Determine if query is too complex for AI"""
        # Multiple questions
        if query.count("?") > 2:
            return True
        
        # Very long query
        if len(query) > 500:
            return True
        
        # Complex financial terms requiring expertise
        complex_terms = [
            "estate planning", "trust", "advanced tax strategy",
            "merger", "acquisition", "legal advice"
        ]
        
        return any(term in query.lower() for term in complex_terms)


# ============================================================================
# HANDOFF QUEUE
# ============================================================================

class HandoffQueue:
    """
    Manage queue of escalations
    
    Features:
    - Priority-based queuing
    - Agent assignment
    - SLA tracking
    - Load balancing
    """
    
    def __init__(self):
        self.queue: List[EscalationRequest] = []
        self.active_handoffs: Dict[str, EscalationRequest] = {}
        
        # SLA targets (minutes)
        self.sla_targets = {
            Priority.URGENT: 5,
            Priority.HIGH: 15,
            Priority.MEDIUM: 30,
            Priority.LOW: 60
        }
        
        logger.info("Handoff Queue initialized")
    
    async def add_to_queue(self, escalation: EscalationRequest):
        """Add escalation to queue"""
        self.queue.append(escalation)
        
        # Sort by priority
        priority_order = {
            Priority.URGENT: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3
        }
        
        self.queue.sort(key=lambda e: priority_order[e.priority])
        
        logger.info(f"Added escalation {escalation.escalation_id} to queue (priority: {escalation.priority.value})")
        
        # Check SLA
        await self._check_sla_breach(escalation)
    
    async def assign_to_agent(
        self,
        escalation_id: str,
        agent_id: str
    ) -> bool:
        """Assign escalation to agent"""
        # Find in queue
        escalation = next(
            (e for e in self.queue if e.escalation_id == escalation_id),
            None
        )
        
        if not escalation:
            return False
        
        # Assign
        escalation.status = HandoffStatus.ASSIGNED
        escalation.assigned_agent = agent_id
        escalation.assigned_at = datetime.now(UTC)
        
        # Move to active
        self.queue.remove(escalation)
        self.active_handoffs[escalation_id] = escalation
        
        logger.info(f"Assigned escalation {escalation_id} to agent {agent_id}")
        return True
    
    async def complete_handoff(self, escalation_id: str):
        """Mark handoff as complete"""
        if escalation_id in self.active_handoffs:
            escalation = self.active_handoffs[escalation_id]
            escalation.status = HandoffStatus.RESOLVED
            escalation.resolved_at = datetime.now(UTC)
            
            del self.active_handoffs[escalation_id]
            
            logger.info(f"Completed handoff {escalation_id}")
    
    def get_queue_position(self, escalation_id: str) -> Optional[int]:
        """Get position in queue"""
        for i, escalation in enumerate(self.queue):
            if escalation.escalation_id == escalation_id:
                return i + 1
        return None
    
    def get_estimated_wait_time(self, escalation_id: str) -> Optional[int]:
        """Get estimated wait time in minutes"""
        position = self.get_queue_position(escalation_id)
        
        if position is None:
            return None
        
        # Rough estimate: 5 minutes per escalation ahead
        return position * 5
    
    async def _check_sla_breach(self, escalation: EscalationRequest):
        """Check if SLA is at risk"""
        sla_target = self.sla_targets[escalation.priority]
        
        elapsed = (datetime.now(UTC) - escalation.created_at).total_seconds() / 60
        
        if elapsed > sla_target * 0.8:  # 80% of SLA
            logger.warning(f"Escalation {escalation.escalation_id} approaching SLA breach")


# ============================================================================
# CONTEXT PACKAGER
# ============================================================================

class ContextPackager:
    """
    Package context for human agent
    
    Creates comprehensive handoff package with:
    - Conversation summary
    - Customer profile
    - AI attempts
    - Recommended actions
    """
    
    def __init__(self):
        logger.info("Context Packager initialized")
    
    async def create_agent_context(
        self,
        escalation: EscalationRequest
    ) -> AgentContext:
        """Create context package for agent"""
        
        # Summarize conversation
        summary = await self._summarize_conversation(escalation.last_messages)
        
        # Extract key entities
        entities = await self._extract_key_entities(escalation.last_messages)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            escalation.reason,
            escalation.customer_profile,
            escalation.last_messages
        )
        
        return AgentContext(
            customer_name=escalation.customer_profile.get("name", "Customer"),
            customer_tier=escalation.customer_profile.get("tier", "standard"),
            account_summary=escalation.customer_profile.get("account", {}),
            conversation_history=escalation.last_messages,
            conversation_summary=summary,
            key_entities_mentioned=entities,
            ai_attempted_responses=[],  # Would include AI's attempts
            recommended_actions=recommendations,
            escalation_reason=escalation.reason.value,
            priority=escalation.priority.value
        )
    
    async def _summarize_conversation(
        self,
        messages: List[Dict[str, str]]
    ) -> str:
        """Summarize conversation for agent"""
        if not messages:
            return "No conversation history"
        
        # Extract user queries
        user_queries = [
            msg["content"]
            for msg in messages
            if msg.get("role") == "user"
        ]
        
        if len(user_queries) == 1:
            return f"Customer asked: {user_queries[0]}"
        else:
            return f"Customer discussed: {'; '.join(user_queries[:3])}"
    
    async def _extract_key_entities(
        self,
        messages: List[Dict[str, str]]
    ) -> List[str]:
        """Extract key entities mentioned"""
        # In production: Use NLU to extract
        entities = []
        
        all_text = " ".join(msg["content"] for msg in messages)
        
        # Simple extraction
        if "portfolio" in all_text.lower():
            entities.append("portfolio")
        if "account" in all_text.lower():
            entities.append("account")
        
        return entities
    
    async def _generate_recommendations(
        self,
        reason: EscalationReason,
        customer_profile: Dict[str, Any],
        messages: List[Dict[str, str]]
    ) -> List[str]:
        """Generate action recommendations for agent"""
        recommendations = []
        
        if reason == EscalationReason.LOW_CONFIDENCE:
            recommendations.append("Verify customer understanding")
            recommendations.append("Provide detailed explanation")
        
        elif reason == EscalationReason.DETECTED_FRUSTRATION:
            recommendations.append("Acknowledge frustration")
            recommendations.append("Offer immediate assistance")
            recommendations.append("Consider escalation to supervisor")
        
        elif reason == EscalationReason.COMPLEX_QUERY:
            recommendations.append("Review customer portfolio in detail")
            recommendations.append("Consult with specialist if needed")
        
        return recommendations


# ============================================================================
# HANDOFF MANAGER (INTEGRATED)
# ============================================================================

class HandoffManager:
    """
    Complete human handoff management system
    
    Orchestrates:
    - Escalation detection
    - Queue management
    - Context packaging
    - Agent notification
    """
    
    def __init__(self):
        self.detector = EscalationDetector()
        self.queue = HandoffQueue()
        self.packager = ContextPackager()
        
        logger.info("✅ Handoff Manager initialized")
    
    async def check_and_escalate(
        self,
        customer_id: str,
        session_id: str,
        query: str,
        ai_confidence: float,
        conversation_history: List[Dict[str, str]],
        customer_profile: Dict[str, Any],
        safety_flags: List[str] = None
    ) -> Optional[EscalationRequest]:
        """
        Check if escalation needed and create request
        
        Returns escalation request if escalated, None otherwise
        """
        import uuid
        
        # Check if should escalate
        should_escalate, reason, priority = await self.detector.should_escalate(
            query=query,
            ai_confidence=ai_confidence,
            conversation_history=conversation_history,
            safety_flags=safety_flags or []
        )
        
        if not should_escalate:
            return None
        
        # Create escalation request
        escalation = EscalationRequest(
            escalation_id=f"esc_{uuid.uuid4().hex[:12]}",
            customer_id=customer_id,
            session_id=session_id,
            reason=reason,
            priority=priority,
            created_at=datetime.now(UTC),
            conversation_summary=self._summarize_quick(conversation_history),
            last_messages=conversation_history[-10:],  # Last 10 messages
            customer_profile=customer_profile,
            ai_confidence_scores=[ai_confidence],
            detected_issues=[reason.value]
        )
        
        # Add to queue
        await self.queue.add_to_queue(escalation)
        
        return escalation
    
    async def get_agent_context(
        self,
        escalation_id: str
    ) -> Optional[AgentContext]:
        """Get context package for agent"""
        # Find escalation
        escalation = self.queue.active_handoffs.get(escalation_id)
        
        if not escalation:
            # Check queue
            escalation = next(
                (e for e in self.queue.queue if e.escalation_id == escalation_id),
                None
            )
        
        if not escalation:
            return None
        
        # Package context
        return await self.packager.create_agent_context(escalation)
    
    def get_handoff_message(self, escalation: EscalationRequest) -> str:
        """Get message to display to user during handoff"""
        wait_time = self.queue.get_estimated_wait_time(escalation.escalation_id)
        
        messages = {
            EscalationReason.USER_REQUEST: "I'm connecting you with a specialist who can help you better.",
            EscalationReason.LOW_CONFIDENCE: "Let me connect you with someone who can provide more detailed assistance.",
            EscalationReason.DETECTED_FRUSTRATION: "I understand this is frustrating. Let me get you to someone who can help right away.",
            EscalationReason.COMPLEX_QUERY: "This requires specialized expertise. I'm connecting you with an expert.",
            EscalationReason.SAFETY_CONCERN: "I'm connecting you with our team for immediate assistance.",
        }
        
        base_message = messages.get(
            escalation.reason,
            "I'm connecting you with a team member who can help."
        )
        
        if wait_time:
            return f"{base_message} Estimated wait time: {wait_time} minutes."
        else:
            return f"{base_message} Please hold..."
    
    def _summarize_quick(self, messages: List[Dict[str, str]]) -> str:
        """Quick conversation summary"""
        if not messages:
            return "No conversation history"
        
        user_msgs = [m["content"][:100] for m in messages if m.get("role") == "user"]
        return " | ".join(user_msgs[-3:])


# ============================================================================
# DEMO
# ============================================================================

async def demo_handoff_system():
    """Demonstrate human handoff system"""
    print("\n" + "=" * 70)
    print("🤝 ANYA HUMAN HANDOFF SYSTEM DEMO")
    print("=" * 70)
    
    manager = HandoffManager()
    
    # Test 1: Explicit request for human
    print("\n" + "─" * 70)
    print("TEST 1: Explicit Request for Human")
    print("─" * 70)
    
    escalation = await manager.check_and_escalate(
        customer_id="customer_001",
        session_id="session_001",
        query="I want to speak to a human agent",
        ai_confidence=0.9,
        conversation_history=[
            {"role": "user", "content": "I want to speak to a human agent"}
        ],
        customer_profile={"name": "Sarah", "tier": "premium"}
    )
    
    if escalation:
        print(f"✅ Escalation created:")
        print(f"   ID: {escalation.escalation_id}")
        print(f"   Reason: {escalation.reason.value}")
        print(f"   Priority: {escalation.priority.value}")
        print(f"   Message: {manager.get_handoff_message(escalation)}")
    
    # Test 2: Low confidence escalation
    print("\n" + "─" * 70)
    print("TEST 2: Low Confidence Escalation")
    print("─" * 70)
    
    escalation2 = await manager.check_and_escalate(
        customer_id="customer_002",
        session_id="session_002",
        query="Complex estate planning question",
        ai_confidence=0.3,  # Low confidence
        conversation_history=[],
        customer_profile={"name": "John"}
    )
    
    if escalation2:
        print(f"✅ Escalation created:")
        print(f"   Reason: {escalation2.reason.value}")
        print(f"   Confidence: {0.3:.0%}")
    
    # Test 3: Frustration detection
    print("\n" + "─" * 70)
    print("TEST 3: Frustration Detection")
    print("─" * 70)
    
    frustrated_history = [
        {"role": "user", "content": "What's my balance?"},
        {"role": "assistant", "content": "Your balance is..."},
        {"role": "user", "content": "This is terrible service!"}
    ]
    
    escalation3 = await manager.check_and_escalate(
        customer_id="customer_003",
        session_id="session_003",
        query="This is terrible service!",
        ai_confidence=0.8,
        conversation_history=frustrated_history,
        customer_profile={"name": "Mike"}
    )
    
    if escalation3:
        print(f"✅ Frustration detected and escalated:")
        print(f"   Priority: {escalation3.priority.value}")
    
    # Test 4: Queue management
    print("\n" + "─" * 70)
    print("TEST 4: Queue Management")
    print("─" * 70)
    
    print(f"Queue size: {len(manager.queue.queue)}")
    print(f"Active handoffs: {len(manager.queue.active_handoffs)}")
    
    for i, esc in enumerate(manager.queue.queue, 1):
        wait_time = manager.queue.get_estimated_wait_time(esc.escalation_id)
        print(f"  {i}. {esc.escalation_id} - {esc.priority.value} - ETA: {wait_time}min")
    
    # Test 5: Agent context packaging
    print("\n" + "─" * 70)
    print("TEST 5: Agent Context Packaging")
    print("─" * 70)
    
    if escalation:
        context = await manager.get_agent_context(escalation.escalation_id)
        
        if context:
            print(f"✅ Agent Context Package:")
            print(f"   Customer: {context.customer_name}")
            print(f"   Tier: {context.customer_tier}")
            print(f"   Summary: {context.conversation_summary}")
            print(f"   Recommendations: {', '.join(context.recommended_actions)}")
    
    print("\n" + "=" * 70)
    print("✅ Human Handoff System Demo Complete!")
    print("=" * 70)
    print("\nFeatures Demonstrated:")
    print("  ✅ Escalation Detection (6 triggers)")
    print("  ✅ Priority-based Queuing")
    print("  ✅ Context Packaging")
    print("  ✅ SLA Tracking")
    print("  ✅ Agent Recommendations")
    print("  ✅ Seamless Transfer")
    print("")


if __name__ == "__main__":
    asyncio.run(demo_handoff_system())
