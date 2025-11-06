# Australian Tax Compliance - CORRECTED

## ⚠️ CRITICAL: ATO Wash Sale Rules

**I previously stated Australia has no wash sale rules - THIS WAS INCORRECT.**

### The Truth About Australian Wash Sales

While Australia doesn't have a specific named "wash sale rule" like the USA, **the ATO actively enforces anti-avoidance provisions** under:

- **Part IVA** - General Anti-Avoidance Rule (Section 177D, 177F)
- **TR 2008/1** - ATO Taxation Ruling on Wash Sales

### What the ATO Considers a Wash Sale

1. **Selling an asset to realize a capital loss**
2. **Repurchasing same or substantially identical asset**
3. **Within a short timeframe (especially <30 days)**
4. **Primary purpose is to obtain tax benefit**
5. **No genuine change in economic exposure**

### Penalties

- ❌ **Capital loss disallowed**
- ⚠️ **Penalties up to 50% of tax avoided**
- 📊 **Interest charges**
- 🔍 **Potential audit of other transactions**

---

## ✅ Compliant Strategies

### Strategy 1: 45-Day Waiting Period
**SAFEST APPROACH**

- Sell asset on Day 0
- Wait 45+ calendar days
- Repurchase on Day 46+
- **Risk: LOW** - Demonstrates genuine disposition

**Example:**
\\\
Sell VAS.AX: June 1, 2025
Earliest safe rebuy: July 16, 2025 (45 days later)
\\\

### Strategy 2: Asset Switching
**RECOMMENDED FOR IMMEDIATE REPLACEMENT**

Sell one asset, immediately buy substantially different but similar exposure:

**ASX 200 Alternatives:**
- VAS.AX → A200.AX (Different fund manager)
- VAS.AX → STW.AX (Different methodology)
- VAS.AX → IOZ.AX (Different index provider)

**International Alternatives:**
- VGS.AX → IWLD.AX
- VGS.AX → WXOZ.AX
- VGS.AX → VGAD.AX (hedged version)

**Risk: LOW** - Genuinely different securities

### Strategy 3: Genuine Rebalancing
Document as part of legitimate portfolio management:

- Written investment policy
- Regular rebalancing schedule
- Asset allocation targets
- Risk management rationale

**Risk: LOW to MEDIUM** - Must be genuine, documented strategy

---

## API Endpoints

### 1. Calculate CGT (Corrected)
\\\
POST /api/v1/tax/calculate-cgt

Body:
{
    "purchase_date": "2023-06-01",
    "sale_date": "2025-06-15",
    "cost_base": 50000,
    "proceeds": 65000
}

Response:
{
    "capital_gain": 15000,
    "cgt_discount": 0.5,
    "taxable_gain": 7500,
    "wash_sale_risk": {
        "high_risk_until": "2025-07-15",
        "safe_to_rebuy_after": "2025-07-30",
        "warning": "ATO monitors under Part IVA",
        "recommendation": "Wait 45+ days OR switch asset"
    }
}
\\\

### 2. Check Wash Sale Risk
\\\
POST /api/v1/tax/check-wash-sale-risk

Body:
{
    "asset_id": "VAS.AX",
    "sale_date": "2025-06-15",
    "proposed_rebuy_date": "2025-06-20",
    "is_loss": true,
    "has_offsetting_gains": true
}

Response:
{
    "risk_assessment": {
        "risk_level": "HIGH",
        "risk_score": 9,
        "days_between_trades": 5,
        "explanation": "HIGH RISK: ATO likely to challenge"
    },
    "recommendations": [{
        "strategy": "WAIT",
        "action": "Wait 40 more days"
    }, {
        "strategy": "SWITCH_ASSET",
        "alternatives": ["A200.AX", "STW.AX"]
    }]
}
\\\

### 3. Optimize Tax Harvesting (Compliant)
\\\
POST /api/v1/tax/optimize-harvest

Body:
{
    "positions": [{
        "asset_id": "VAS.AX",
        "unrealized_gain": -5000,
        "holding_days": 200
    }],
    "target_harvest_date": "2025-06-15"
}

Response:
{
    "recommendations": [{
        "asset": "VAS.AX",
        "action": "HARVEST_LOSS",
        "strategy": {
            "option_1": {
                "method": "WAIT_AND_REBUY",
                "timeline": "Sell now, rebuy after 2025-07-30",
                "risk": "LOW"
            },
            "option_2": {
                "method": "IMMEDIATE_SWITCH",
                "alternatives": ["A200.AX", "STW.AX"],
                "risk": "LOW"
            },
            "option_3": {
                "method": "RISKY_REBUY",
                "risk": "HIGH",
                "warning": "NOT RECOMMENDED"
            }
        },
        "recommendation": "Use Option 2 for best compliance"
    }]
}
\\\

### 4. Suggest Alternatives
\\\
POST /api/v1/tax/suggest-alternatives

Body:
{
    "asset_id": "VAS.AX"
}

Response:
{
    "alternatives": [{
        "ticker": "A200.AX",
        "substantially_different": true,
        "similar_exposure": true
    }],
    "recommendation": "Sell VAS.AX, immediately buy A200.AX"
}
\\\

### 5. Compliance Check
\\\
POST /api/v1/tax/compliance-check

Body:
{
    "transactions": [{
        "date": "2025-06-15",
        "type": "sell",
        "asset_id": "VAS.AX",
        "is_loss": true
    }, {
        "date": "2025-06-20",
        "type": "buy",
        "asset_id": "VAS.AX"
    }]
}

Response:
{
    "compliance_status": "FAILED",
    "issues": [{
        "severity": "HIGH",
        "issue": "Potential wash sale",
        "recommendation": "Wait 45+ days OR switch asset"
    }]
}
\\\

---

## Testing

\\\powershell
# Start API
cd capsules\src\services\capsule_service
python app.py

# Look for: "✓ Australian Tax Service registered (ATO Part IVA Compliant)"

# Test 1: CGT Calculation
$body = '{"purchase_date":"2023-06-01","sale_date":"2025-06-15","cost_base":50000,"proceeds":65000}'
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tax/calculate-cgt" -Method POST -ContentType "application/json" -Body $body | ConvertTo-Json

# Test 2: Wash Sale Risk Check
$body = '{"asset_id":"VAS.AX","sale_date":"2025-06-15","proposed_rebuy_date":"2025-06-20","is_loss":true,"has_offsetting_gains":true}'
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tax/check-wash-sale-risk" -Method POST -ContentType "application/json" -Body $body | ConvertTo-Json

# Test 3: Suggest Alternatives
$body = '{"asset_id":"VAS.AX"}'
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tax/suggest-alternatives" -Method POST -ContentType "application/json" -Body $body | ConvertTo-Json
\\\

---

## Key Takeaways

1. ✅ **Australia DOES monitor wash sales** under Part IVA
2. ⏱️ **45+ day waiting period** is best practice
3. 🔄 **Asset switching** is effective and compliant
4. 📝 **Document everything** - genuine investment rationale
5. ⚠️ **High risk period** is <30 days, especially near June 30
6. 💰 **Penalties are severe** - up to 50% of tax avoided

---

## Compliance Checklist

Before harvesting any capital loss:

- [ ] Have I waited 45+ days before repurchasing?
- [ ] OR am I switching to substantially different asset?
- [ ] Is this part of documented investment strategy?
- [ ] Have I kept records of transaction rationale?
- [ ] Am I avoiding transactions purely for tax benefit?
- [ ] Is the timing suspicious (near June 30)?

---

**Version:** 3.1 (CORRECTED)  
**Status:** ATO Part IVA Compliant  
**Updated:** 2025-11-07 10:09:32  
**Legislation:** ITAA 1997 Part IVA, TR 2008/1
