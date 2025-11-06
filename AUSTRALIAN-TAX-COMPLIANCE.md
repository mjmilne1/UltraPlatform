# Australian Tax Compliance - Capsules Platform

## 🇦🇺 Overview

Complete Australian Taxation Office (ATO) compliance with:
- **Capital Gains Tax (CGT)** - 50% discount for assets held >12 months
- **Franking Credits** - Dividend imputation credits
- **No Wash Sale Rules** - Can immediately rebuy after selling (unlike USA)

---

## Key Features

### 1. CGT 50% Discount
**Benefit:** Pay tax on only 50% of capital gains  
**Requirement:** Hold asset >12 months (individuals)

**Example:**
- Purchase: $50,000
- Sale: $65,000  
- Gain: $15,000
- Held: 13 months → 50% discount applies
- Taxable: $7,500 (save ~$3,525 at 47% rate)

### 2. No Wash Sale Rules
**USA:** Can't rebuy for 30 days after loss  
**Australia:** Can rebuy immediately ✅

**Strategy:**
1. Sell asset to realize loss
2. Immediately rebuy same asset
3. Maintain market exposure
4. Optimize taxes

### 3. Franking Credits
**Benefit:** Avoid double taxation on dividends  
**Result:** Can receive tax refund

**Example:**
- Dividend: $1,000
- Franking credits: $429 (100% franked)
- If marginal rate < 30%: Get refund!

---

## API Endpoints

### Calculate CGT
\\\
POST /api/v1/tax/calculate-cgt

Body:
{
    "purchase_date": "2023-06-01",
    "sale_date": "2024-08-15",
    "cost_base": 50000,
    "proceeds": 65000
}

Response:
{
    "capital_gain": 15000,
    "holding_days": 440,
    "cgt_discount": 0.5,
    "discount_amount": 7500,
    "taxable_gain": 7500,
    "can_rebuy_immediately": true
}
\\\

### Optimize Tax Harvesting
\\\
POST /api/v1/tax/optimize-harvest

Body:
{
    "positions": [{
        "asset_id": "VAS.AX",
        "unrealized_gain": -5000,
        "holding_days": 200
    }]
}

Response:
{
    "recommendations": [{
        "action": "HARVEST_LOSS",
        "loss_amount": 5000,
        "tax_benefit": 2350,
        "strategy": "Sell now, rebuy immediately"
    }]
}
\\\

### Calculate Franking Credits
\\\
POST /api/v1/tax/franking-credits

Body:
{
    "dividend": 1000,
    "franking_percentage": 100,
    "marginal_tax_rate": 0.37
}

Response:
{
    "dividend": 1000,
    "franking_credits": 428.57,
    "net_benefit": 128.57,
    "is_refund": true
}
\\\

---

## Database Tables

### tax_lots
Tracks cost base for FIFO CGT calculation

### cgt_events
Records all capital gains transactions

---

## Testing

\\\powershell
# Start API
cd capsules\src\services\capsule_service
python app.py

# Test CGT (in new window)
$body = '{"purchase_date":"2023-06-01","sale_date":"2024-08-15","cost_base":50000,"proceeds":65000}'
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tax/calculate-cgt" -Method POST -ContentType "application/json" -Body $body
\\\

---

## Compliance

✅ **ATO CGT Rules** - ITAA 1997  
✅ **Franking Credits** - Dividend imputation system  
✅ **FIFO Method** - First-in-first-out cost base  
✅ **Financial Year** - July 1 to June 30  

---

**Version:** 3.0.0  
**Status:** Production Ready  
**Compliance:** Australian Taxation Office  
**Upgraded:** 2025-11-07 10:02:30
