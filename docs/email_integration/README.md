# TuringWealth - Email Integration (Turing Dynamics Edition)

## ?? Overview

AI-powered email service with:
- Multi-provider support (SendGrid, AWS SES, SMTP)
- Rich HTML templates with Jinja2
- AI-powered content generation
- ML personalization engine
- Email analytics & tracking
- Transactional & marketing emails
- DataMesh integration
- MCP control interface

## ?? Quick Start

### 1. Configure Provider

**SendGrid (Recommended):**
```bash
export SENDGRID_API_KEY="your-api-key"
```

**AWS SES:**
```bash
export AWS_ACCESS_KEY="your-access-key"
export AWS_SECRET_KEY="your-secret-key"
```

### 2. Run Database Migration
```bash
mysql -u root -p < backend/migrations/email/001_create_email_tables.sql
```

### 3. Initialize Service
```python
from app.integrations.email.email_service import EmailService, EmailProvider

email = EmailService(
    db_session,
    datamesh_client,
    provider=EmailProvider.SENDGRID,
    api_key=os.getenv("SENDGRID_API_KEY")
)

await email.initialize()
```

### 4. Send Your First Email
```python
await email.send_email(
    to=["client@example.com"],
    subject="Welcome to TuringWealth",
    body_html="<h1>Welcome!</h1><p>We're excited to have you.</p>",
    priority=EmailPriority.HIGH
)
```

## ?? Email Types

### Transactional Emails
- Welcome emails
- Password resets
- Account notifications
- Transaction confirmations

### Report Emails
- Monthly statements
- Performance reports
- Tax summaries
- Custom reports

### Approval Emails
- Approval requests
- Approval confirmations
- Rejection notifications

### Alert Emails
- Critical alerts
- Compliance violations
- System notifications
- Security alerts

### Marketing Emails
- Newsletters
- Product updates
- Educational content
- Campaigns

## ?? Using Templates

### Send from Template
```python
await email.send_from_template(
    template_name="monthly_statement",
    to=["client@example.com"],
    template_data={
        "client_name": "John Smith",
        "month": "November",
        "year": 2025,
        "opening_balance": "$1,200,000",
        "closing_balance": "$1,250,000",
        "return": "+4.2%",
        "pdf_url": "https://..."
    }
)
```

### Available Templates
- `welcome` - New client welcome
- `report_ready` - Report notification
- `monthly_statement` - Monthly statement
- `approval_request` - Approval needed
- `alert` - Critical alert

### Create Custom Template
```python
template = EmailTemplate(
    template_id="custom_report",
    template_name="Custom Report",
    email_type=EmailType.REPORT,
    subject_template="Your {{ report_type }} Report - {{ client_name }}",
    html_template="<h1>Report for {{ client_name }}</h1>...",
    text_template="Report for {{ client_name }}..."
)
```

## ?? AI Features

### AI Content Generation
```python
from app.integrations.email.agents import ContentGenerationAgent

agent = ContentGenerationAgent(email, memory)

# Generate personalized subject
subject = await agent.generate_personalized_subject(
    recipient_name="John Smith",
    email_type="monthly_statement",
    context={"month": "November"}
)
```

### ML Send Time Optimization
```python
# Automatically determine best send time
optimal_time = await agent.optimize_send_time(
    recipient_email="client@example.com",
    email_type="report"
)
# Returns: 09:00 (based on recipient's open history)
```

### Personalization
```python
# ML-based personalization
prediction = personalization_model.predict_engagement(
    recipient_profile={
        "name": "John",
        "active_user": True,
        "engagement_score": 0.85
    },
    email_context={
        "email_type": "report",
        "send_time": "09:00"
    }
)
# Returns: engagement_score, open_probability, click_probability
```

## ?? Pre-built Email Functions

### Welcome Email
```python
await email.send_welcome_email(
    client_email="newclient@example.com",
    client_name="Jane Doe"
)
```

### Report Email
```python
await email.send_report_email(
    client_email="client@example.com",
    client_name="John Smith",
    report_name="Monthly Statement - November 2025",
    report_url="https://app.turingwealth.com/reports/...",
    report_type="monthly_statement"
)
```

### Approval Notification
```python
await email.send_approval_notification(
    approver_email="approver@turingwealth.com",
    approver_name="Sarah Manager",
    request_id="req-123",
    request_type="Client Account Creation",
    request_details={
        "client": "ABC Corp",
        "amount": "$100,000"
    },
    approval_url="https://app.turingwealth.com/approvals/req-123"
)
```

### Alert Email
```python
await email.send_alert_email(
    recipient_email="risk@turingwealth.com",
    alert_title="High Risk Transaction Detected",
    alert_message="Transaction TXN-123 flagged for review",
    severity="critical",
    action_url="https://app.turingwealth.com/transactions/123"
)
```

## ?? Bulk Emails

Send personalized bulk emails:
```python
await email.send_bulk_email(
    template_name="newsletter",
    recipients=[
        {
            "email": "client1@example.com",
            "data": {"name": "John", "content": "..."}
        },
        {
            "email": "client2@example.com",
            "data": {"name": "Jane", "content": "..."}
        }
    ],
    priority=EmailPriority.LOW
)
```

## ?? Analytics & Tracking

### Email Analytics
```python
# Get analytics via MCP
analytics = await mcp_client.call_tool("get_email_analytics", {
    "days": 30
})

# Returns:
{
    "total_sent": 1250,
    "delivered": 1230,
    "opened": 850,
    "clicked": 320,
    "bounced": 20,
    "delivery_rate": 98.4%,
    "open_rate": 68.0%,
    "click_rate": 25.6%
}
```

### Track Opens
```python
# Automatic tracking via pixel
# When recipient opens email, webhook fires:
await email.track_open(message_id="msg-123")
```

### Track Clicks
```python
# Automatic link tracking
# When recipient clicks link, webhook fires:
await email.track_click(
    message_id="msg-123",
    url="https://app.turingwealth.com/reports"
)
```

## ?? MCP Tools

Control email via MCP:
```python
# Send email
await mcp_client.call_tool("send_email", {
    "to": ["client@example.com"],
    "subject": "Welcome",
    "body_html": "<h1>Welcome!</h1>"
})

# Send from template
await mcp_client.call_tool("send_from_template", {
    "template_name": "welcome",
    "to": ["client@example.com"],
    "template_data": {"client_name": "John"}
})

# Get status
await mcp_client.call_tool("get_email_status", {
    "message_id": "msg-123"
})
```

## ?? Email Priorities

- **LOW** - Marketing, newsletters (batched)
- **MEDIUM** - Reports, notifications (normal queue)
- **HIGH** - Transactional, approvals (priority queue)
- **URGENT** - Alerts, critical (immediate send)

## ?? Security & Compliance

### Bounce Handling
- **Hard Bounce** - Automatically unsubscribe
- **Soft Bounce** - Retry up to 3 times
- **Complaint** - Immediate unsubscribe + admin notification

### Unsubscribe
- One-click unsubscribe
- Global unsubscribe list
- GDPR compliant
- Automatic link injection

### Privacy
- No PII in tracking pixels
- Encrypted storage
- GDPR compliance
- Audit trail logging

## ?? DataMesh Integration

All email events published to DataMesh:
```python
# Events:
- EMAIL_QUEUED
- EMAIL_SENT
- EMAIL_DELIVERED
- EMAIL_OPENED
- EMAIL_CLICKED
- EMAIL_BOUNCED
- EMAIL_FAILED
```

## ?? Performance

- **Queue Processing** - 50 emails/batch
- **Sending Rate** - 100 emails/minute
- **Delivery Rate** - 98%+
- **Open Rate** - 65%+ (transactional)
- **Response Time** - <5 seconds to queue

## ??? Providers

### SendGrid (Recommended)
- Easy setup
- Excellent deliverability
- Built-in analytics
- 100 emails/day free

### AWS SES
- Cost-effective at scale
- High reliability
- AWS ecosystem integration
- $0.10 per 1,000 emails

### SMTP
- Self-hosted option
- Full control
- No external dependencies
- Requires MTA setup

## ?? Support

- Documentation: `/docs/email_integration`
- Issues: GitHub issues
- Email: support@turingwealth.com

## ? Checklist

Setup checklist:
- [ ] Choose provider (SendGrid/SES/SMTP)
- [ ] Configure API keys
- [ ] Run database migration
- [ ] Load email templates
- [ ] Test sending
- [ ] Configure tracking
- [ ] Set up webhooks
- [ ] Enable analytics
- [ ] Configure bounce handling
- [ ] Test unsubscribe flow
