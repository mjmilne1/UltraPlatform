package com.ultra.platform.service;

import com.ultra.platform.integration.LedgerIntegrationService;
import com.ultra.platform.entity.Portfolio;
import com.ultra.platform.entity.Position;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.util.*;

@Service
@RequiredArgsConstructor
@Slf4j
public class PortfolioService {
    
    private final LedgerIntegrationService ledgerService;
    
    public Portfolio createPortfolio(String clientName, BigDecimal initialDeposit) {
        // Create account in UltraLedger2
        UUID ledgerAccountId = ledgerService.createInvestmentAccount(
            clientName, "USD"
        );
        
        // Record initial deposit
        ledgerService.recordInvestmentTransaction(
            ledgerAccountId,
            initialDeposit,
            "DEPOSIT",
            "Initial portfolio funding"
        );
        
        Portfolio portfolio = Portfolio.builder()
            .id(UUID.randomUUID())
            .clientName(clientName)
            .ledgerAccountId(ledgerAccountId)
            .cashBalance(initialDeposit)
            .totalValue(initialDeposit)
            .positions(new ArrayList<>())
            .createdAt(new Date())
            .build();
        
        log.info("Created portfolio for {} with ledger account {}", 
                 clientName, ledgerAccountId);
        
        return portfolio;
    }
    
    public void executeTrade(UUID portfolioId, String symbol, 
                            int quantity, BigDecimal price, String type) {
        BigDecimal tradeValue = price.multiply(BigDecimal.valueOf(quantity));
        
        // Get portfolio's ledger account
        UUID ledgerAccountId = getPortfolioLedgerAccount(portfolioId);
        
        // Record trade in UltraLedger2
        String description = String.format("%s %d shares of %s @ %s", 
                                          type, quantity, symbol, price);
        
        ledgerService.recordInvestmentTransaction(
            ledgerAccountId,
            type.equals("BUY") ? tradeValue.negate() : tradeValue,
            "TRADE",
            description
        );
        
        // Update cash balance from ledger
        BigDecimal newBalance = ledgerService.getAccountBalance(ledgerAccountId);
        updatePortfolioCashBalance(portfolioId, newBalance);
        
        log.info("Executed trade: {}", description);
    }
    
    public BigDecimal calculatePortfolioValue(UUID portfolioId, String asOfDate) {
        UUID ledgerAccountId = getPortfolioLedgerAccount(portfolioId);
        
        // Get historical cash balance from UltraLedger2
        BigDecimal cashBalance = ledgerService.getHistoricalBalance(
            ledgerAccountId, asOfDate
        );
        
        // Add position values
        BigDecimal positionValue = calculatePositionValues(portfolioId, asOfDate);
        
        return cashBalance.add(positionValue);
    }
    
    private UUID getPortfolioLedgerAccount(UUID portfolioId) {
        // Retrieve ledger account ID for portfolio
        return UUID.randomUUID(); // Placeholder
    }
    
    private void updatePortfolioCashBalance(UUID portfolioId, BigDecimal balance) {
        // Update portfolio cash balance
        log.info("Updated portfolio {} cash balance to {}", portfolioId, balance);
    }
    
    private BigDecimal calculatePositionValues(UUID portfolioId, String asOfDate) {
        // Calculate total value of all positions
        return BigDecimal.ZERO; // Placeholder
    }
}
