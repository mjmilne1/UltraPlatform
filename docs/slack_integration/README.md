# TuringWealth - Slack Integration (Turing Dynamics Edition)

## ?? Overview

AI-powered Slack integration with:
- Real-time webhook notifications
- Interactive slash commands (/tw)
- AI-powered message routing
- ML intent classification
- Interactive buttons & modals
- DataMesh event streaming
- Agentic AI responses

## ?? Quick Start

### 1. Create Slack App

1. Go to https://api.slack.com/apps
2. Click "Create New App" ? "From scratch"
3. Name: "TuringWealth Bot"
4. Choose your workspace

### 2. Configure Bot Permissions

Add these scopes:
- `chat:write` - Send messages
- `commands` - Use slash commands
- `channels:read` - Read channel info
- `users:read` - Read user info
- `im:write` - Send DMs

### 3. Install to Workspace

1. Install app to workspace
2. Copy Bot User OAuth Token
3. Add to environment:
```bash
export SLACK_BOT_TOKEN="xoxb-your-token"
export SLACK_SIGNING_SECRET="your-secret"
```

### 4. Set Up Slash Command

1. Go to Slash Commands
2. Create `/tw` command
3. Request URL: `https://your-domain.com/api/slack/commands`
4. Save

### 5. Initialize Integration
```python
from app.integrations.slack.slack_integration_service import SlackIntegrationService

slack = SlackIntegrationService(
    db_session,
    datamesh_client,
    bot_token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET")
)

await slack.initialize()
```

## ?? Features

### Real-Time Notifications

Send notifications to any channel:
```python
await slack.send_notification(
    channel="#operations",
    title="COB Process Complete",
    text="Close of Business process completed successfully",
    priority=SlackPriority.MEDIUM
)
```

### Critical Alerts with Actions

Send alerts with interactive buttons:
```python
await slack.send_alert(
    channel="#alerts",
    title="High Risk Transaction Detected",
    text="Transaction TXN-12345 flagged as high risk ($250,000)",
    severity="critical",
    action_buttons=[
        {"text": "Approve", "action": "approve_txn_12345"},
        {"text": "Reject", "action": "reject_txn_12345"}
    ]
)
```

### Approval Requests

Send maker-checker approval requests:
```python
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
```

### Report Notifications

Notify when reports are ready:
```python
await slack.send_report_notification(
    channel="#reports",
    report_name="Monthly Statement - November 2025",
    report_type="monthly_statement",
    download_url="https://app.turingwealth.com/reports/...",
    generated_for="Client ABC"
)
```

## ?? Slash Commands

### Available Commands

**Portfolio Management:**
```
/tw portfolio ABC Corp       # Show portfolio summary
/tw performance ABC Corp     # Show performance metrics
```

**Reporting:**
```
/tw report monthly ABC Corp      # Generate monthly statement
/tw report performance ABC Corp  # Generate performance report
```

**Approvals:**
```
/tw approve req-123    # Approve pending request
/tw reject req-123     # Reject pending request
/tw pending            # Show pending approvals
```

**System:**
```
/tw status    # System status
/tw cob       # COB process status
/tw help      # Show help
```

**Admin:**
```
/tw sync              # Trigger data sync
/tw audit client-123  # Show audit trail
```

## ?? AI Features

### Intent Classification

Automatically classifies user intent:
```python
# User types: "/tw show me the portfolio for ABC Corp"

intent = await slack._classify_intent("show me the portfolio for ABC Corp")
# Result:
{
    "intent": "portfolio",
    "confidence": 0.92,
    "entities": {
        "client_id": "ABC Corp"
    }
}
```

### Smart Message Routing

AI routes messages to appropriate teams:
```python
# Incoming message: "Urgent: System is down!"

routing = await routing_agent.process_incoming_message(message)
# Routes to: #alerts
# Notifies: @oncall, @manager
# Priority: critical
```

### Auto-Response to FAQs

Automatically responds to common questions:
```
User: "What are your office hours?"
Bot: "?? Office Hours
     Monday - Friday: 9:00 AM - 5:00 PM AEST
     After hours: support@turingwealth.com"
```

## ?? Message Formatting

### Rich Blocks

Create beautiful, interactive messages:
```python
blocks = [
    {
        "type": "header",
        "text": {"type": "plain_text", "text": "?? Portfolio Update"}
    },
    {
        "type": "section",
        "fields": [
            {"type": "mrkdwn", "text": "*Total Value:* $1,250,000"},
            {"type": "mrkdwn", "text": "*Day Change:* +$12,500 (+1.01%)"}
        ]
    },
    {
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "View Details"},
                "url": "https://app.turingwealth.com/portfolio"
            }
        ]
    }
]
```

### Priority Styling

Automatic emoji based on priority:
- **Low:** ?? Informational
- **Medium:** ?? Notification
- **High:** ?? Warning
- **Critical:** ?? Alert

## ?? Notification Types

### Operational Notifications
- COB process complete
- Job execution status
- Data sync complete
- Scheduled task results

### Alerts
- High-risk transactions
- Compliance violations
- System errors
- Security incidents

### Approvals
- Client onboarding
- Large transactions
- Account changes
- Fee structure changes

### Reports
- Monthly statements ready
- Performance reports ready
- Tax summaries ready
- Custom reports ready

### Compliance
- Audit trail events
- Maker-checker violations
- SOD breaches
- Policy violations

## ?? MCP Tools

Control Slack via MCP:
```python
# Send notification via MCP
await mcp_client.call_tool("send_notification", {
    "channel": "#operations",
    "title": "Report Generated",
    "text": "Monthly report is ready",
    "priority": "medium"
})
```

Available MCP tools:
- `send_notification`
- `send_alert`
- `send_approval_request`
- `send_report_notification`
- `get_message_history`
- `get_command_history`

## ?? Analytics

### Message Analytics

Track message delivery and engagement:
- Total messages sent
- Delivery success rate
- Channel usage stats
- Priority distribution

### Command Analytics

Monitor slash command usage:
- Command frequency
- Intent classification accuracy
- Response times
- Success rates

### Daily Summary

Automated daily summary sent to #operations:
```
?? Daily Slack Summary - Nov 10, 2025

Messages Sent: 147
Commands Executed: 42
Alerts Triggered: 3
Approval Requests: 8

Top Commands:
1. /tw portfolio (15)
2. /tw status (12)
3. /tw report (8)
```

## ?? Security

### Signature Verification

All requests verified using signing secret:
```python
# Automatic signature verification
if not slack._verify_signature(request_headers, request_body):
    return {"error": "Invalid signature"}
```

### Permission Checks

Commands require appropriate permissions:
```python
# /tw approve requires "approver" role
if not user_has_permission(user_id, "approver"):
    return {"error": "Unauthorized"}
```

### Rate Limiting

Prevent abuse:
- 60 messages per minute (total)
- 10 commands per user per minute
- Automatic throttling

## ?? DataMesh Integration

All Slack events published to DataMesh:
```python
# Events published:
- SLACK_MESSAGE_SENT
- SLACK_COMMAND_RECEIVED
- SLACK_INTERACTION_PROCESSED
- SLACK_ALERT_TRIGGERED
- SLACK_APPROVAL_GRANTED
- SLACK_APPROVAL_REJECTED
```

## ?? Integration Examples

### COB Complete Notification
```python
# When COB completes
await slack.send_notification(
    channel="#operations",
    title="COB Process Complete",
    text=f"""
    ? Trade settlements: 150 trades
    ? Portfolio valuation: 250 clients
    ? Fee calculation: Complete
    ? Reconciliation: No discrepancies
    ? Ledger close: Complete
    
    Duration: 18 minutes
    """,
    priority=SlackPriority.MEDIUM
)
```

### High Risk Alert
```python
# High-risk transaction detected
await slack.send_alert(
    channel="#alerts",
    title="High Risk Transaction Detected",
    text=f"""
    *Transaction:* TXN-12345
    *Client:* ABC Corp
    *Amount:* $250,000
    *Risk Score:* 0.87 (High)
    *Reason:* Unusual transaction size
    """,
    severity="high",
    action_buttons=[
        {"text": "Approve", "action": "approve_txn_12345"},
        {"text": "Review", "action": "review_txn_12345"},
        {"text": "Reject", "action": "reject_txn_12345"}
    ]
)
```

### Compliance Violation
```python
# Maker-checker violation
await slack.send_notification(
    channel="#compliance",
    title="Maker-Checker Violation Detected",
    text=f"""
    ?? *Compliance Issue*
    
    *Type:* Self-approval attempt
    *User:* john@turingwealth.com
    *Request:* req-123
    *Time:* {datetime.now()}
    
    Action has been blocked automatically.
    """,
    priority=SlackPriority.HIGH,
    message_type=SlackMessageType.COMPLIANCE_ISSUE
)
```

## ?? Support

- Documentation: `/docs/slack_integration`
- Issues: GitHub issues
- Slack: #turingwealth-support

## ? Checklist

Setup checklist:
- [ ] Create Slack app
- [ ] Configure bot permissions
- [ ] Install to workspace
- [ ] Set up slash command
- [ ] Configure webhooks
- [ ] Test notifications
- [ ] Test slash commands
- [ ] Configure channels
- [ ] Enable AI features
- [ ] Set up DataMesh events
