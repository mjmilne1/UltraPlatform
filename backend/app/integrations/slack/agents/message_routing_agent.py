"""
AI Message Routing Agent
Intelligent routing and response generation for Slack messages
"""

from typing import Dict, List
from datetime import datetime

class MessageRoutingAgent:
    """
    Autonomous agent for intelligent message routing
    
    Capabilities:
    - Intent classification
    - Smart routing to teams
    - Automated responses
    - Escalation handling
    - Context awareness
    """
    
    def __init__(self, slack_service, memory):
        self.slack = slack_service
        self.memory = memory
        self.name = "MessageRoutingAgent"
    
    async def process_incoming_message(
        self,
        message: Dict
    ) -> Dict:
        """
        Process incoming Slack message
        
        Analyzes intent and routes appropriately
        """
        
        print(f"[{self.name}] Processing message from {message['user']}")
        
        # Classify intent
        intent = await self._classify_message_intent(message["text"])
        
        # Determine urgency
        urgency = await self._assess_urgency(message["text"], intent)
        
        # Route message
        routing = await self._determine_routing(intent, urgency)
        
        # Generate response
        response = await self._generate_response(message, intent, routing)
        
        print(f"[{self.name}] Intent: {intent['category']}, Routing: {routing['destination']}")
        
        return {
            "intent": intent,
            "urgency": urgency,
            "routing": routing,
            "response": response
        }
    
    async def _classify_message_intent(self, text: str) -> Dict:
        """
        Classify message intent using ML
        
        Categories:
        - question (needs information)
        - request (needs action)
        - complaint (needs escalation)
        - report (informational)
        - urgent (needs immediate attention)
        """
        
        text_lower = text.lower()
        
        # Simple keyword-based (would use ML model)
        if any(word in text_lower for word in ["urgent", "emergency", "asap", "critical"]):
            return {
                "category": "urgent",
                "confidence": 0.95,
                "keywords": ["urgent", "critical"]
            }
        elif any(word in text_lower for word in ["how", "what", "when", "where", "why", "?"]):
            return {
                "category": "question",
                "confidence": 0.85,
                "keywords": ["question"]
            }
        elif any(word in text_lower for word in ["please", "can you", "could you", "need"]):
            return {
                "category": "request",
                "confidence": 0.80,
                "keywords": ["request"]
            }
        elif any(word in text_lower for word in ["problem", "issue", "error", "broken", "not working"]):
            return {
                "category": "complaint",
                "confidence": 0.85,
                "keywords": ["issue", "problem"]
            }
        else:
            return {
                "category": "report",
                "confidence": 0.60,
                "keywords": []
            }
    
    async def _assess_urgency(
        self,
        text: str,
        intent: Dict
    ) -> str:
        """
        Assess message urgency
        
        Levels: low, medium, high, critical
        """
        
        if intent["category"] == "urgent":
            return "critical"
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["urgent", "asap", "immediately"]):
            return "high"
        elif intent["category"] == "complaint":
            return "high"
        elif intent["category"] == "request":
            return "medium"
        else:
            return "low"
    
    async def _determine_routing(
        self,
        intent: Dict,
        urgency: str
    ) -> Dict:
        """
        Determine where to route message
        
        Routes to:
        - #support (questions)
        - #operations (requests)
        - #alerts (urgent)
        - #compliance (compliance issues)
        """
        
        category = intent["category"]
        
        if urgency == "critical":
            return {
                "destination": "#alerts",
                "notify": ["@oncall", "@manager"],
                "action": "escalate"
            }
        elif category == "question":
            return {
                "destination": "#support",
                "notify": [],
                "action": "respond"
            }
        elif category == "request":
            return {
                "destination": "#operations",
                "notify": ["@ops-team"],
                "action": "assign"
            }
        elif category == "complaint":
            return {
                "destination": "#support",
                "notify": ["@support-lead"],
                "action": "escalate"
            }
        else:
            return {
                "destination": "#general",
                "notify": [],
                "action": "acknowledge"
            }
    
    async def _generate_response(
        self,
        message: Dict,
        intent: Dict,
        routing: Dict
    ) -> str:
        """
        Generate AI-powered response
        
        Creates contextual, helpful responses
        """
        
        category = intent["category"]
        
        if category == "urgent":
            return (
                f"?? Urgent message received. "
                f"I've notified {', '.join(routing['notify'])}. "
                f"Someone will respond immediately."
            )
        elif category == "question":
            return (
                f"Thanks for your question! "
                f"I've routed this to the support team in {routing['destination']}. "
                f"You'll get a response shortly."
            )
        elif category == "request":
            return (
                f"? Request received. "
                f"I've created a ticket and notified the operations team. "
                f"I'll keep you updated on progress."
            )
        elif category == "complaint":
            return (
                f"I'm sorry you're experiencing issues. "
                f"I've escalated this to our support lead who will reach out shortly."
            )
        else:
            return "Message received. Thanks for letting us know!"
    
    async def auto_respond_to_common_queries(
        self,
        message: Dict
    ) -> Dict:
        """
        Automatically respond to common questions
        
        FAQ responses without human intervention
        """
        
        text_lower = message["text"].lower()
        
        # Office hours
        if "office hours" in text_lower or "opening hours" in text_lower:
            return {
                "auto_respond": True,
                "response": (
                    "?? *Office Hours*\n"
                    "Monday - Friday: 9:00 AM - 5:00 PM AEST\n"
                    "After hours support available via email: support@turingwealth.com"
                )
            }
        
        # System status
        if "system status" in text_lower or "is the system down" in text_lower:
            return {
                "auto_respond": True,
                "response": (
                    "? *System Status*\n"
                    "All systems operational ?\n"
                    "Check real-time status: https://status.turingwealth.com"
                )
            }
        
        # Contact info
        if "contact" in text_lower or "phone number" in text_lower:
            return {
                "auto_respond": True,
                "response": (
                    "?? *Contact Information*\n"
                    "Phone: 1300 XXX XXX\n"
                    "Email: support@turingwealth.com\n"
                    "Address: Sydney, NSW"
                )
            }
        
        return {"auto_respond": False}
