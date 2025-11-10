# TuringWealth - Reporting Engine (Turing Dynamics Edition)

## ?? Overview

AI-powered reporting system with:
- 20+ professional report templates
- Multi-format export (PDF, Excel, CSV, HTML)
- AI-powered insights & recommendations
- ML-based performance attribution
- Scheduled report generation
- Real-time data from DataMesh
- Interactive dashboards

## ?? Quick Start

### 1. Run Database Migration
```bash
mysql -u root -p < backend/migrations/reports/001_create_reports_tables.sql
```

### 2. Initialize Engine
```python
from app.reports.reporting_engine import ReportingEngine

reports = ReportingEngine(db_session, datamesh_client)
await reports.initialize()
```

### 3. Generate Your First Report
```python
from datetime import date

# Generate monthly statement
statement = await reports.generate_monthly_statement(
    client_id="client-123",
    month=11,
    year=2025
)
```

## ?? Available Reports

### Client Reports
- **Monthly Statement** - Professional monthly account statement
- **Quarterly Statement** - Comprehensive quarterly review
- **Annual Statement** - Complete annual summary
- **Performance Summary** - Detailed performance analysis with ML attribution
- **Portfolio Valuation** - Real-time portfolio valuation
- **Transaction History** - Complete transaction log
- **Tax Summary** - Australian tax summary (CGT, dividends, franking)

### Management Reports
- **AUM Report** - Assets under management analysis
- **Fee Revenue Report** - Management fee revenue tracking
- **Client Activity** - Client engagement metrics
- **Trading Summary** - Trading activity summary

### Compliance Reports
- **Audit Trail Report** - Complete audit trail
- **Compliance Summary** - Regulatory compliance status
- **Risk Report** - Portfolio risk analysis

### Financial Reports
- **Balance Sheet** - Statement of financial position
- **Income Statement** - P&L statement
- **Cash Flow Statement** - Cash flow analysis
- **Trial Balance** - Accounting trial balance

## ?? AI Features

### AI-Powered Insights
```python
# Automatically generated insights in reports
insights = [
    {
        "type": "STRONG_PERFORMANCE",
        "title": "Strong Outperformance",
        "message": "Portfolio outperformed benchmark by 2.5% this period",
        "sentiment": "positive",
        "priority": "high"
    },
    {
        "type": "CONCENTRATION_RISK",
        "title": "Concentration Risk",
        "message": "Largest holding represents 18% of portfolio",
        "sentiment": "warning",
        "priority": "medium"
    }
]
```

### ML Performance Attribution
```python
attribution = await reports._ml_performance_attribution(
    client_id="client-123",
    portfolio_data=data
)

# Results:
{
    "asset_allocation_effect": 0.015,  # 1.5%
    "security_selection_effect": 0.012,  # 1.2%
    "timing_effect": -0.003,  # -0.3%
    "currency_effect": 0.002,  # 0.2%
    "interaction_effect": 0.004  # 0.4%
}
```

### Personalized Recommendations
- Risk-aware suggestions
- Portfolio rebalancing recommendations
- Opportunity identification
- Market context analysis

## ?? Report Formats

### PDF Export
- Professional corporate branding
- Charts and graphs
- Interactive table of contents
- Digital signatures

### Excel Export
- Multiple worksheets
- Formatted tables
- Embedded formulas
- Interactive charts

### CSV Export
- Raw data export
- Easy integration
- Bulk analysis

### HTML Export
- Interactive dashboards
- Responsive design
- Embeddable widgets

## ?? Scheduled Reports

### Schedule Recurring Reports
```python
await reports_mcp.schedule_report(
    report_type="monthly_statement",
    schedule_pattern="0 0 1 * *",  # 1st of every month
    client_id="client-123",
    format="pdf"
)
```

### Default Schedules
- **Monthly Statements** - 1st of each month
- **Quarterly Reports** - 1st of Jan, Apr, Jul, Oct
- **Annual Reports** - July 1 (Australian FY)

## ?? MCP Tools

- `generate_report` - Generate any report type
- `generate_monthly_statement` - Monthly client statement
- `generate_performance_report` - Performance analysis
- `generate_tax_summary` - Australian tax summary
- `get_report_status` - Check generation status
- `schedule_report` - Schedule recurring report
- `get_report_history` - View past reports

## ?? Professional Features

### Monthly Statement Includes:
- Portfolio summary (opening/closing balance)
- Performance metrics (returns, benchmark comparison)
- Holdings breakdown (current positions)
- Transaction history (all activity)
- Fee summary (management fees)
- AI insights (opportunities, risks, recommendations)

### Tax Summary Includes:
- Capital gains/losses
- CGT discount (50% for 12+ months)
- Dividend income
- Franking credits
- Interest income
- Deductible investment fees

### Performance Report Includes:
- Time-weighted returns (TWR)
- Money-weighted returns (MWR)
- Benchmark comparison (ASX 200, ASX 300)
- ML performance attribution
- Risk metrics (volatility, Sharpe, max drawdown)
- AI recommendations

## ?? Customization

### Custom Templates
```python
# Create custom report template
template = {
    "template_name": "Custom Monthly",
    "report_type": "monthly_statement",
    "sections": ["summary", "holdings", "insights"],
    "branding": {
        "logo": "custom_logo.png",
        "primary_color": "#1E40AF"
    }
}
```

### Branding Options
- Company logo
- Color schemes
- Headers/footers
- Disclaimers
- Watermarks

## ?? Performance

- **Concurrent Generation** - Up to 5 reports simultaneously
- **Caching** - 1-hour cache for repeated requests
- **Async Processing** - Background generation via Celery
- **Optimization** - ML-optimized templates

## ?? Security

- Report access control (role-based)
- Encrypted storage
- Audit trail logging
- Watermarking
- Digital signatures

## ?? Analytics

### Report Usage Analytics
```python
# Track report generation
{
    "total_generated": 1250,
    "by_type": {
        "monthly_statement": 800,
        "performance_summary": 250,
        "tax_summary": 200
    },
    "avg_generation_time": 3.5,  # seconds
    "success_rate": 0.995
}
```

## ?? DataMesh Integration

All reports integrate with DataMesh for:
- Real-time data streaming
- Event publication
- State synchronization
- Historical data access

## ?? Support

- Documentation: `/docs/reports_module`
- Issues: GitHub issues
- Slack: #turingwealth-reports

## ? Next Steps

1. ? Run database migration
2. ? Configure branding
3. ? Set up scheduled reports
4. ? Test report generation
5. ? Enable AI insights
6. ? Configure delivery (email)
