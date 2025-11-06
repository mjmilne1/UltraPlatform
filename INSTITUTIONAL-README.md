# Capsules Platform - Institutional Edition v2.0.0

## ASIC RG 255 Compliant Onboarding System

### 🎯 Overview
Your Capsules Platform has been upgraded with institutional-grade onboarding and risk profiling capabilities.

### ✅ New Features
- **ASIC RG 255 Compliant** - Automated digital advice framework
- **Risk Profiling** - 5-question Likert scale psychometric assessment
- **KYC Tables** - Identity verification and compliance
- **Audit Trail** - Complete regulatory logging
- **Session Management** - Multi-step onboarding workflow

### 🔌 API Endpoints

#### Onboarding
- **POST** `/api/v1/onboarding/start` - Start new session
- **POST** `/api/v1/onboarding/<id>/risk-assessment` - Submit risk questionnaire
- **GET** `/api/v1/onboarding/<id>/status` - Get session status
- **GET** `/api/v1/onboarding/health` - Health check

### 📊 Risk Assessment Scoring

**5-Question Likert Scale (1-5):**
- **R1:** Volatility comfort
- **R2:** Loss tolerance
- **R3:** Capital protection (reverse-scored)
- **R4:** Buy-the-dip mentality
- **R5:** Drawdown anxiety (reverse-scored)

**Risk Bands:**
- **Conservative:** ≤ 11 points
- **Balanced:** 12-17 points
- **Growth:** ≥ 18 points

### 🚀 Quick Start

1. Start the API:
```bash
cd capsules\src\services\capsule_service
python app.py
```

2. Test the endpoint:
```bash
curl -X POST http://localhost:8000/api/v1/onboarding/start -H "Content-Type: application/json" -d "{}"
```

3. Submit risk assessment:
```bash
curl -X POST http://localhost:8000/api/v1/onboarding/session_test/risk-assessment \
  -H "Content-Type: application/json" \
  -d '{"r1_score":4,"r2_score":4,"r3_score":2,"r4_score":3,"r5_score":3}'
```

### 📋 Database Tables
1. **client_kyc** - Identity verification
2. **risk_assessment** - Risk profiling data
3. **onboarding_sessions** - Workflow state
4. **audit_log** - Compliance trail

### 🧪 Testing

See `test-institutional.ps1` for comprehensive test suite.

### 📝 Compliance
- ASIC RG 255 - Automated advice
- AUSTRAC RG 8 - Identity verification
- Data retention: 7 years

---
**Upgraded:** 2025-11-07 08:11:24  
**Version:** 2.0.0  
**Status:** Production Ready
