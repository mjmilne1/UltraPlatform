# UltraPlatform - Enterprise Investment Management Platform

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/mjmilne1/UltraPlatform)
[![UltraLedger2](https://img.shields.io/badge/Powered%20by-UltraLedger2-blue)](https://github.com/mjmilne1/UltraLedger2)

Enterprise-grade investment management platform powered by UltraLedger2's bank-grade financial ledger infrastructure.

## ??? Architecture
$platformReadme = @'
# UltraPlatform - Enterprise Investment Management Platform

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/mjmilne1/UltraPlatform)
[![UltraLedger2](https://img.shields.io/badge/Powered%20by-UltraLedger2-blue)](https://github.com/mjmilne1/UltraLedger2)

Enterprise-grade investment management platform powered by UltraLedger2's bank-grade financial ledger infrastructure.

## ??? Architecture
```
+--------------------------------------+
¦       UltraPlatform (Port 8081)      ¦
¦   Investment Management Services      ¦
+--------------------------------------¦
¦      LedgerIntegrationService        ¦
¦         REST API Calls ?             ¦
+--------------------------------------¦
¦      UltraLedger2 (Port 8080)        ¦
¦   Bank-Grade Financial Ledger        ¦
¦   • Bitemporal Event Sourcing        ¦
¦   • ACID Transactions                ¦
¦   • Complete Audit Trail             ¦
+--------------------------------------+
```

## ?? Quick Start

### Prerequisites
- Java 17+
- Maven 3.6+
- PostgreSQL 14+ (or Docker)
- UltraLedger2 running on port 8080

### Installation

1. **Start UltraLedger2** (Financial Ledger)
```bash
cd ../UltraLedger2
./mvnw spring-boot:run
# Verify at http://localhost:8080/swagger-ui.html
```

2. **Start UltraPlatform** (Investment Platform)
```bash
cd UltraPlatform
./mvnw spring-boot:run
# Access at http://localhost:8081
```

## ? Features

### Investment Management
- **Portfolio Management** - Multi-asset portfolios
- **Trade Execution** - Buy/sell with ledger recording
- **Position Tracking** - Real-time position management
- **Performance Analytics** - Returns and risk metrics

### Financial Infrastructure (via UltraLedger2)
- **Double-Entry Bookkeeping** - Every transaction balanced
- **Bitemporal Data** - Complete audit history
- **Event Sourcing** - Full transaction replay
- **ACID Guarantees** - Transaction integrity

## ?? API Endpoints

### Portfolio Operations
- `POST /api/v1/portfolios` - Create portfolio
- `GET /api/v1/portfolios/{id}` - Get portfolio details
- `POST /api/v1/portfolios/{id}/trades` - Execute trade
- `GET /api/v1/portfolios/{id}/value` - Calculate portfolio value

### Integration with UltraLedger2
- All financial transactions recorded in UltraLedger2
- Temporal queries for historical valuations
- Complete audit trail for compliance

## ?? Dependencies

This platform requires UltraLedger2 running as the financial ledger backend:
- Repository: https://github.com/mjmilne1/UltraLedger2
- Default URL: http://localhost:8080

## ?? Use Cases

1. **Portfolio Creation**
   - Creates investment account in UltraLedger2
   - Records initial deposit
   - Tracks all subsequent transactions

2. **Trade Execution**
   - Records trade in ledger
   - Updates cash balance
   - Maintains position records

3. **Historical Valuation**
   - Uses bitemporal queries
   - Reconstructs portfolio state
   - Provides audit trail

## ??? Technology Stack

- **Spring Boot 3.2** - Application framework
- **UltraLedger2** - Financial ledger backend
- **PostgreSQL** - Database
- **RestTemplate** - Service integration
- **Maven** - Build management

## ?? License

MIT License

## ?? Author

**mjmilne1**
- UltraPlatform: [@mjmilne1/UltraPlatform](https://github.com/mjmilne1/UltraPlatform)
- UltraLedger2: [@mjmilne1/UltraLedger2](https://github.com/mjmilne1/UltraLedger2)

---

*Powered by UltraLedger2 - Bank-grade financial infrastructure*
