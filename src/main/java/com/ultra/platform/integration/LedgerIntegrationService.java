package com.ultra.platform.integration;

import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import java.math.BigDecimal;
import java.util.Map;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class LedgerIntegrationService {
    
    private final RestTemplate restTemplate = new RestTemplate();
    
    @Value("${ultraledger.api.url:http://localhost:8080}")
    private String ledgerApiUrl;
    
    // Create investment account in UltraLedger2
    public UUID createInvestmentAccount(String clientName, String currency) {
        String url = ledgerApiUrl + "/api/v1/accounts";
        
        Map<String, Object> request = Map.of(
            "accountNumber", "INV-" + System.nanoTime(),
            "customerName", clientName,
            "currency", currency,
            "type", "INVESTMENT"
        );
        
        ResponseEntity<Map> response = restTemplate.postForEntity(url, request, Map.class);
        log.info("Created investment account for {}", clientName);
        
        return UUID.fromString((String) response.getBody().get("id"));
    }
    
    // Record investment transaction in ledger
    public void recordInvestmentTransaction(UUID accountId, 
                                           BigDecimal amount, 
                                           String type,
                                           String description) {
        String url = ledgerApiUrl + "/api/v1/transactions/transfer";
        
        Map<String, Object> request = Map.of(
            "accountId", accountId,
            "amount", amount,
            "type", type,
            "description", description,
            "timestamp", System.currentTimeMillis()
        );
        
        restTemplate.postForEntity(url, request, Map.class);
        log.info("Recorded {} transaction: {} for account {}", type, amount, accountId);
    }
    
    // Get account balance from ledger
    public BigDecimal getAccountBalance(UUID accountId) {
        String url = ledgerApiUrl + "/api/v1/accounts/" + accountId + "/balance";
        ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
        return new BigDecimal(response.getBody().get("balance").toString());
    }
    
    // Transfer between investment accounts
    public void transferFunds(UUID fromAccount, UUID toAccount, 
                             BigDecimal amount, String description) {
        String url = ledgerApiUrl + "/api/v1/transactions/transfer";
        
        Map<String, Object> request = Map.of(
            "fromAccountId", fromAccount,
            "toAccountId", toAccount,
            "amount", amount,
            "description", "Investment: " + description
        );
        
        restTemplate.postForEntity(url, request, Map.class);
        log.info("Transferred {} from {} to {}", amount, fromAccount, toAccount);
    }
    
    // Query temporal balance for portfolio valuation
    public BigDecimal getHistoricalBalance(UUID accountId, String asOfDate) {
        String url = ledgerApiUrl + "/api/v1/events/balance/temporal"
                   + "?accountId=" + accountId 
                   + "&asOf=" + asOfDate;
        
        ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
        return new BigDecimal(response.getBody().get("balance").toString());
    }
}
