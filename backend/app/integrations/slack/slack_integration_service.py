"""
TuringWealth - Slack Integration Service (Turing Dynamics Edition)
AI-powered team collaboration and notifications
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import uuid
import asyncio
import httpx
import json

# ============================================================================
# DOMAIN MODELS
# ============================================================================

class SlackMessageType(Enum):
    NOTIFICATION = "notification"
    ALERT = "alert"
    APPROVAL_REQUEST = "approval_request"
    REPORT_READY = "report_ready"
    COMPLIANCE_ISSUE = "compliance_issue"
    SYSTEM_STATUS = "system_status"

class SlackPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CommandCategory(Enum):
    REPORTING = "reporting"
    PORTFOLIO = "portfolio"
    COMPLIANCE = "compliance"
    SYSTEM = "system"
    ADMIN = "admin"

@dataclass
class SlackWorkspace:
    """Slack workspace connection"""
    workspace_id: str
    team_name: str
    bot_token: str
    app_token: str
    webhook_url: str
    
    connected_at: datetime
    last_activity_at: Optional[datetime] = None
    active: bool = True

@dataclass
class SlackMessage:
    """Slack message to send"""
    message_id: str
    channel: str
    message_type: SlackMessageType
    priority: SlackPriority
    
    # Content
    title: str
    text: str
    blocks: Optional[List[Dict]] = None
    attachments: Optional[List[Dict]] = None
    
    # Metadata
    created_at: datetime = None
    sent_at: Optional[datetime] = None
    thread_ts: Optional[str] = None  # For threading
    
    # Status
    status: str = "pending"
    error_message: Optional[str] = None

@dataclass
class SlackCommand:
    """Slash command received"""
    command_id: str
    command: str  # e.g., "/tw"
    text: str  # Command parameters
    
    user_id: str
    user_name: str
    channel_id: str
    team_id: str
    
    trigger_id: str  # For opening modals
    response_url: str  # For delayed responses
    
    received_at: datetime
    processed_at: Optional[datetime] = None

class SlackIntegrationService:
    """
    Slack Integration Service
    
    Features:
    - Real-time webhook notifications
    - Interactive slash commands
    - AI-powered message routing
    - ML intent classification
    - Interactive buttons & modals
    - Event streaming to DataMesh
    - Agentic AI responses
    """
    
    def __init__(
        self,
        db_session,
        datamesh_client=None,
        mcp_client=None,
        bot_token: str = None,
        signing_secret: str = None
    ):
        self.db = db_session
        self.datamesh = datamesh_client
        self.mcp = mcp_client
        
        self.bot_token = bot_token
        self.signing_secret = signing_secret
        self.api_base = "https://slack.com/api"
        
        self.active_workspace: Optional[SlackWorkspace] = None
        self.message_queue: List[SlackMessage] = []
    
    async def initialize(self):
        """Initialize Slack connection"""
        
        # Load workspace configuration
        workspace = await self._load_workspace()
        
        if workspace:
            self.active_workspace = workspace
            print(f"? Connected to Slack: {workspace.team_name}")
            
            # Start message processing
            asyncio.create_task(self._process_message_queue())
        else:
            print("? No Slack workspace configured")
    
    async def send_notification(
        self,
        channel: str,
        title: str,
        text: str,
        priority: SlackPriority = SlackPriority.MEDIUM,
        message_type: SlackMessageType = SlackMessageType.NOTIFICATION,
        blocks: Optional[List[Dict]] = None,
        thread_ts: Optional[str] = None
    ) -> Dict:
        """
        Send notification to Slack channel
        
        Args:
            channel: Channel name (e.g., "#operations")
            title: Message title
            text: Message body
            priority: Message priority
            message_type: Type of notification
            blocks: Rich message blocks
            thread_ts: Thread to reply to
        """
        
        message_id = str(uuid.uuid4())
        
        # Create message
        message = SlackMessage(
            message_id=message_id,
            channel=channel,
            message_type=message_type,
            priority=priority,
            title=title,
            text=text,
            blocks=blocks or self._create_default_blocks(title, text, priority),
            created_at=datetime.now(),
            thread_ts=thread_ts
        )
        
        # Queue for processing
        self.message_queue.append(message)
        
        # Publish to DataMesh
        if self.datamesh:
            await self.datamesh.events.publish({
                "event_type": "SLACK_MESSAGE_QUEUED",
                "message_id": message_id,
                "channel": channel,
                "priority": priority.value,
                "timestamp": datetime.now().isoformat()
            })
        
        return {"message_id": message_id, "status": "queued"}
    
    async def send_alert(
        self,
        channel: str,
        title: str,
        text: str,
        severity: str = "high",
        action_buttons: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Send critical alert with optional action buttons
        
        Example:
            await slack.send_alert(
                channel="#alerts",
                title="High Risk Transaction Detected",
                text="Transaction TXN-123 flagged as high risk",
                severity="critical",
                action_buttons=[
                    {"text": "Approve", "action": "approve_txn_123"},
                    {"text": "Reject", "action": "reject_txn_123"}
                ]
            )
        """
        
        blocks = self._create_alert_blocks(title, text, severity, action_buttons)
        
        priority = SlackPriority.CRITICAL if severity == "critical" else SlackPriority.HIGH
        
        return await self.send_notification(
            channel=channel,
            title=title,
            text=text,
            priority=priority,
            message_type=SlackMessageType.ALERT,
            blocks=blocks
        )
    
    async def send_approval_request(
        self,
        channel: str,
        request_id: str,
        title: str,
        details: Dict,
        approvers: List[str]
    ) -> Dict:
        """
        Send approval request with interactive buttons
        
        Example:
            await slack.send_approval_request(
                channel="#approvals",
                request_id="req-123",
                title="Client Account Creation",
                details={
                    "client": "ABC Corp",
                    "account_type": "Investment Account",
                    "initial_deposit": "$100,000"
                },
                approvers=["@john", "@sarah"]
            )
        """
        
        blocks = self._create_approval_blocks(request_id, title, details, approvers)
        
        return await self.send_notification(
            channel=channel,
            title=title,
            text=f"Approval required: {title}",
            priority=SlackPriority.HIGH,
            message_type=SlackMessageType.APPROVAL_REQUEST,
            blocks=blocks
        )
    
    async def send_report_notification(
        self,
        channel: str,
        report_name: str,
        report_type: str,
        download_url: str,
        generated_for: str
    ) -> Dict:
        """
        Notify about completed report
        
        Example:
            await slack.send_report_notification(
                channel="#reports",
                report_name="Monthly Statement - November 2025",
                report_type="monthly_statement",
                download_url="https://app.turingwealth.com/reports/...",
                generated_for="Client ABC"
            )
        """
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "?? Report Ready"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{report_name}*\n\n"
                            f"*Type:* {report_type}\n"
                            f"*For:* {generated_for}\n"
                            f"*Generated:* {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Download Report"
                        },
                        "url": download_url,
                        "style": "primary"
                    }
                ]
            }
        ]
        
        return await self.send_notification(
            channel=channel,
            title=f"Report Ready: {report_name}",
            text=f"Report {report_name} is ready for download",
            priority=SlackPriority.MEDIUM,
            message_type=SlackMessageType.REPORT_READY,
            blocks=blocks
        )
    
    async def handle_slash_command(
        self,
        command_data: Dict
    ) -> Dict:
        """
        Handle slash command (e.g., /tw)
        
        Commands:
        - /tw help - Show available commands
        - /tw portfolio [client] - Show portfolio summary
        - /tw report [type] [client] - Generate report
        - /tw approve [request_id] - Approve request
        - /tw status - System status
        """
        
        command_id = str(uuid.uuid4())
        
        # Parse command
        command = SlackCommand(
            command_id=command_id,
            command=command_data["command"],
            text=command_data["text"],
            user_id=command_data["user_id"],
            user_name=command_data["user_name"],
            channel_id=command_data["channel_id"],
            team_id=command_data["team_id"],
            trigger_id=command_data.get("trigger_id", ""),
            response_url=command_data["response_url"],
            received_at=datetime.now()
        )
        
        # Save command
        await self._save_command(command)
        
        # AI-powered intent classification
        intent = await self._classify_intent(command.text)
        
        print(f"?? Slash command received: {command.command} {command.text}")
        print(f"   Classified intent: {intent['intent']} (confidence: {intent['confidence']})")
        
        # Route to handler
        response = await self._route_command(command, intent)
        
        # Mark as processed
        command.processed_at = datetime.now()
        await self._save_command(command)
        
        # Publish to DataMesh
        if self.datamesh:
            await self.datamesh.events.publish({
                "event_type": "SLACK_COMMAND_PROCESSED",
                "command_id": command_id,
                "intent": intent["intent"],
                "timestamp": datetime.now().isoformat()
            })
        
        return response
    
    async def handle_interaction(
        self,
        interaction_data: Dict
    ) -> Dict:
        """
        Handle interactive component (button click, modal submission)
        
        Example interactions:
        - Approval button clicks
        - Report download clicks
        - Form submissions
        """
        
        interaction_type = interaction_data["type"]
        
        if interaction_type == "block_actions":
            # Button click
            action = interaction_data["actions"][0]
            action_id = action["action_id"]
            
            print(f"?? Button clicked: {action_id}")
            
            # Route to handler
            if action_id.startswith("approve_"):
                return await self._handle_approval_action(interaction_data, "approve")
            elif action_id.startswith("reject_"):
                return await self._handle_approval_action(interaction_data, "reject")
            
        elif interaction_type == "view_submission":
            # Modal submission
            view = interaction_data["view"]
            callback_id = view["callback_id"]
            
            print(f"?? Modal submitted: {callback_id}")
            
            return await self._handle_modal_submission(interaction_data)
        
        return {"response_action": "clear"}
    
    async def open_modal(
        self,
        trigger_id: str,
        title: str,
        callback_id: str,
        blocks: List[Dict]
    ) -> Dict:
        """
        Open interactive modal
        
        Example:
            await slack.open_modal(
                trigger_id=trigger_id,
                title="Generate Report",
                callback_id="generate_report",
                blocks=[
                    {
                        "type": "input",
                        "label": "Report Type",
                        "element": {"type": "static_select", ...}
                    }
                ]
            )
        """
        
        modal_view = {
            "type": "modal",
            "callback_id": callback_id,
            "title": {
                "type": "plain_text",
                "text": title
            },
            "submit": {
                "type": "plain_text",
                "text": "Submit"
            },
            "blocks": blocks
        }
        
        response = await self._call_slack_api(
            "views.open",
            {
                "trigger_id": trigger_id,
                "view": modal_view
            }
        )
        
        return response
    
    async def _process_message_queue(self):
        """Process queued messages"""
        
        while True:
            if self.message_queue:
                message = self.message_queue.pop(0)
                
                try:
                    # Send to Slack
                    await self._send_message(message)
                    
                    message.status = "sent"
                    message.sent_at = datetime.now()
                    
                    print(f"? Sent Slack message: {message.message_id}")
                    
                except Exception as e:
                    message.status = "failed"
                    message.error_message = str(e)
                    
                    print(f"? Failed to send message: {str(e)}")
                
                # Save status
                await self._save_message(message)
            
            await asyncio.sleep(1)
    
    async def _send_message(self, message: SlackMessage):
        """Send message to Slack"""
        
        payload = {
            "channel": message.channel,
            "blocks": message.blocks,
            "text": message.text  # Fallback text
        }
        
        if message.thread_ts:
            payload["thread_ts"] = message.thread_ts
        
        response = await self._call_slack_api("chat.postMessage", payload)
        
        message.thread_ts = response.get("ts")
    
    async def _route_command(
        self,
        command: SlackCommand,
        intent: Dict
    ) -> Dict:
        """Route command to appropriate handler"""
        
        intent_type = intent["intent"]
        
        if intent_type == "help":
            return await self._handle_help_command(command)
        elif intent_type == "portfolio":
            return await self._handle_portfolio_command(command, intent)
        elif intent_type == "report":
            return await self._handle_report_command(command, intent)
        elif intent_type == "approve":
            return await self._handle_approve_command(command, intent)
        elif intent_type == "status":
            return await self._handle_status_command(command)
        else:
            return {
                "response_type": "ephemeral",
                "text": "Sorry, I didn't understand that command. Type `/tw help` for available commands."
            }
    
    async def _handle_help_command(self, command: SlackCommand) -> Dict:
        """Show help text"""
        
        help_text = """
*TuringWealth Slack Commands*

*Portfolio Management:*
- `/tw portfolio [client]` - Show portfolio summary
- `/tw performance [client]` - Show performance metrics

*Reporting:*
- `/tw report monthly [client]` - Generate monthly statement
- `/tw report performance [client]` - Generate performance report

*Approvals:*
- `/tw approve [request_id]` - Approve pending request
- `/tw reject [request_id]` - Reject pending request
- `/tw pending` - Show pending approvals

*System:*
- `/tw status` - System status
- `/tw cob` - COB process status
- `/tw help` - Show this help

*Admin:*
- `/tw sync` - Trigger data sync
- `/tw audit [entity_id]` - Show audit trail
        """
        
        return {
            "response_type": "ephemeral",
            "text": help_text
        }
    
    async def _handle_portfolio_command(
        self,
        command: SlackCommand,
        intent: Dict
    ) -> Dict:
        """Handle portfolio command"""
        
        client_id = intent.get("entities", {}).get("client_id")
        
        if not client_id:
            return {
                "response_type": "ephemeral",
                "text": "Please specify a client: `/tw portfolio [client_name]`"
            }
        
        # Get portfolio data (mock)
        portfolio = {
            "client_name": "ABC Corp",
            "total_value": 1250000.00,
            "day_change": 12500.00,
            "day_change_pct": 1.01,
            "ytd_return": 8.5
        }
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"?? Portfolio: {portfolio['client_name']}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Total Value:*\n${portfolio['total_value']:,.2f}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Day Change:*\n+${portfolio['day_change']:,.2f} (+{portfolio['day_change_pct']}%)"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*YTD Return:*\n{portfolio['ytd_return']}%"
                    }
                ]
            }
        ]
        
        return {
            "response_type": "in_channel",
            "blocks": blocks
        }
    
    async def _handle_report_command(
        self,
        command: SlackCommand,
        intent: Dict
    ) -> Dict:
        """Handle report generation command"""
        
        report_type = intent.get("entities", {}).get("report_type", "monthly")
        client_id = intent.get("entities", {}).get("client_id")
        
        # Trigger report generation
        # (Would integrate with reporting engine)
        
        return {
            "response_type": "ephemeral",
            "text": f"? Generating {report_type} report... You'll be notified when it's ready."
        }
    
    async def _handle_approve_command(
        self,
        command: SlackCommand,
        intent: Dict
    ) -> Dict:
        """Handle approval command"""
        
        request_id = intent.get("entities", {}).get("request_id")
        
        if not request_id:
            return {
                "response_type": "ephemeral",
                "text": "Please specify a request ID: `/tw approve [request_id]`"
            }
        
        # Process approval (would integrate with audit system)
        
        return {
            "response_type": "in_channel",
            "text": f"? Request {request_id} approved by <@{command.user_id}>"
        }
    
    async def _handle_status_command(self, command: SlackCommand) -> Dict:
        """Show system status"""
        
        # Get system status (mock)
        status = {
            "api": "operational",
            "database": "operational",
            "cob": "completed",
            "last_cob": "2025-11-09 18:30",
            "active_users": 15
        }
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "? System Status"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*API:* ? {status['api']}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Database:* ? {status['database']}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*COB:* ? {status['cob']}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Last COB:* {status['last_cob']}"
                    }
                ]
            }
        ]
        
        return {
            "response_type": "in_channel",
            "blocks": blocks
        }
    
    async def _handle_approval_action(
        self,
        interaction: Dict,
        action_type: str
    ) -> Dict:
        """Handle approval button click"""
        
        action = interaction["actions"][0]
        action_id = action["action_id"]
        
        # Extract request ID
        request_id = action_id.replace(f"{action_type}_", "")
        
        user = interaction["user"]
        
        # Process approval (integrate with audit system)
        
        # Update message
        return {
            "replace_original": True,
            "text": f"? Request {request_id} {action_type}d by <@{user['id']}>"
        }
    
    async def _handle_modal_submission(self, interaction: Dict) -> Dict:
        """Handle modal form submission"""
        
        view = interaction["view"]
        callback_id = view["callback_id"]
        values = view["state"]["values"]
        
        # Extract form data
        form_data = {}
        for block_id, block_values in values.items():
            for action_id, action_value in block_values.items():
                form_data[action_id] = action_value.get("value") or action_value.get("selected_option")
        
        print(f"?? Form submitted: {form_data}")
        
        return {"response_action": "clear"}
    
    async def _classify_intent(self, text: str) -> Dict:
        """
        AI-powered intent classification
        
        Uses ML to understand user intent
        """
        
        # Simple keyword-based classification (would use ML model)
        text_lower = text.lower()
        
        if not text or text_lower == "help":
            return {"intent": "help", "confidence": 1.0}
        elif "portfolio" in text_lower:
            return {
                "intent": "portfolio",
                "confidence": 0.9,
                "entities": {"client_id": "client-123"}
            }
        elif "report" in text_lower:
            return {
                "intent": "report",
                "confidence": 0.9,
                "entities": {"report_type": "monthly", "client_id": "client-123"}
            }
        elif "approve" in text_lower:
            return {
                "intent": "approve",
                "confidence": 0.95,
                "entities": {"request_id": "req-123"}
            }
        elif "status" in text_lower:
            return {"intent": "status", "confidence": 1.0}
        else:
            return {"intent": "unknown", "confidence": 0.0}
    
    def _create_default_blocks(
        self,
        title: str,
        text: str,
        priority: SlackPriority
    ) -> List[Dict]:
        """Create default message blocks"""
        
        emoji = {
            SlackPriority.LOW: "??",
            SlackPriority.MEDIUM: "??",
            SlackPriority.HIGH: "??",
            SlackPriority.CRITICAL: "??"
        }[priority]
        
        return [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {title}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": text
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"TuringWealth • {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    }
                ]
            }
        ]
    
    def _create_alert_blocks(
        self,
        title: str,
        text: str,
        severity: str,
        action_buttons: Optional[List[Dict]]
    ) -> List[Dict]:
        """Create alert blocks with actions"""
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"?? {title}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Severity:* {severity.upper()}\n\n{text}"
                }
            }
        ]
        
        if action_buttons:
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": btn["text"]
                        },
                        "action_id": btn["action"],
                        "style": "primary" if "approve" in btn["action"] else "danger"
                    }
                    for btn in action_buttons
                ]
            })
        
        return blocks
    
    def _create_approval_blocks(
        self,
        request_id: str,
        title: str,
        details: Dict,
        approvers: List[str]
    ) -> List[Dict]:
        """Create approval request blocks"""
        
        details_text = "\n".join([f"*{k}:* {v}" for k, v in details.items()])
        approvers_text = ", ".join(approvers)
        
        return [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"? Approval Required"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{title}*\n\n{details_text}\n\n*Approvers:* {approvers_text}"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "? Approve"
                        },
                        "action_id": f"approve_{request_id}",
                        "style": "primary"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "? Reject"
                        },
                        "action_id": f"reject_{request_id}",
                        "style": "danger"
                    }
                ]
            }
        ]
    
    async def _call_slack_api(self, method: str, payload: Dict) -> Dict:
        """Call Slack API"""
        
        headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.api_base}/{method}"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                raise Exception(f"Slack API error: {response.text}")
            
            result = response.json()
            
            if not result.get("ok"):
                raise Exception(f"Slack API error: {result.get('error')}")
            
            return result
    
    async def _save_message(self, message: SlackMessage):
        """Persist message record"""
        await self.db.execute("""
            INSERT OR REPLACE INTO slack_messages (
                message_id, channel, message_type, priority,
                title, text, blocks, created_at, sent_at,
                thread_ts, status, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message.message_id, message.channel, message.message_type.value,
            message.priority.value, message.title, message.text,
            json.dumps(message.blocks), message.created_at, message.sent_at,
            message.thread_ts, message.status, message.error_message
        ))
        await self.db.commit()
    
    async def _save_command(self, command: SlackCommand):
        """Persist command record"""
        await self.db.execute("""
            INSERT OR REPLACE INTO slack_commands (
                command_id, command, text, user_id, user_name,
                channel_id, team_id, trigger_id, response_url,
                received_at, processed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            command.command_id, command.command, command.text,
            command.user_id, command.user_name, command.channel_id,
            command.team_id, command.trigger_id, command.response_url,
            command.received_at, command.processed_at
        ))
        await self.db.commit()
    
    async def _load_workspace(self) -> Optional[SlackWorkspace]:
        """Load Slack workspace config"""
        # Would load from database
        return None
