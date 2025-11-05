package com.ultraplatform.banking.risk;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.*;

@Component
@Slf4j
public class LiquidityMonitor {
    
    @Autowired
    private JdbcTemplate jdbcTemplate;
    
    private static final BigDecimal LCR_MINIMUM = new BigDecimal("1.0"); // 100%
    private static final BigDecimal NSFR_MINIMUM = new BigDecimal("1.0"); // 100%
    
    @Scheduled(fixedDelay = 60000) // Every minute for real-time monitoring
    public void monitorLiquidity() {
        LiquidityMetrics metrics = calculateLiquidityMetrics();
        
        // Check Basel III requirements
        checkLCR(metrics.lcr);
        checkNSFR(metrics.nsfr);
        
        // Monitor intraday liquidity
        monitorIntradayLiquidity(metrics);
        
        // Store metrics
        storeLiquidityMetrics(metrics);
    }
    
    public LiquidityMetrics calculateLiquidityMetrics() {
        LiquidityMetrics metrics = new LiquidityMetrics();
        
        // High Quality Liquid Assets (HQLA)
        BigDecimal hqla = calculateHQLA();
        
        // Net cash outflows
        BigDecimal netCashOutflows = calculateNetCashOutflows();
        
        // Liquidity Coverage Ratio (LCR)
        metrics.lcr = netCashOutflows.compareTo(BigDecimal.ZERO) > 0 
            ? hqla.divide(netCashOutflows, 4, RoundingMode.HALF_UP)
            : BigDecimal.ONE;
        
        // Available Stable Funding
        BigDecimal asf = calculateAvailableStableFunding();
        
        // Required Stable Funding
        BigDecimal rsf = calculateRequiredStableFunding();
        
        // Net Stable Funding Ratio (NSFR)
        metrics.nsfr = rsf.compareTo(BigDecimal.ZERO) > 0
            ? asf.divide(rsf, 4, RoundingMode.HALF_UP)
            : BigDecimal.ONE;
        
        // Current liquidity position
        metrics.currentCash = getCurrentCashPosition();
        metrics.availableCredit = getAvailableCreditLines();
        
        // Stress metrics
        metrics.stressedOutflows = calculateStressedOutflows();
        
        return metrics;
    }
    
    private BigDecimal calculateHQLA() {
        // Level 1 assets (cash, central bank reserves)
        String level1SQL = """
            SELECT COALESCE(SUM(balance), 0) 
            FROM accounts 
            WHERE account_type IN ('CASH', 'RESERVE', 'SETTLEMENT')
            AND balance > 0
        """;
        
        BigDecimal level1 = jdbcTemplate.queryForObject(level1SQL, BigDecimal.class);
        
        // Level 2 assets (eligible securities) - simplified
        BigDecimal level2 = level1.multiply(new BigDecimal("0.15")); // Assume 15% additional
        
        return level1.add(level2);
    }
    
    private BigDecimal calculateNetCashOutflows() {
        // Expected outflows over 30 days
        String outflowSQL = """
            SELECT 
                COALESCE(SUM(
                    CASE 
                        WHEN account_type = 'DEMAND' THEN balance * 0.1  -- 10% runoff
                        WHEN account_type = 'SAVINGS' THEN balance * 0.05 -- 5% runoff
                        WHEN account_type = 'TERM' THEN balance * 0.0    -- 0% runoff
                        ELSE balance * 0.25 -- Conservative 25% for others
                    END
                ), 0) as expected_outflow
            FROM accounts
            WHERE balance > 0
        """;
        
        BigDecimal outflows = jdbcTemplate.queryForObject(outflowSQL, BigDecimal.class);
        
        // Expected inflows (75% of contractual inflows)
        String inflowSQL = """
            SELECT COALESCE(SUM(amount) * 0.75, 0)
            FROM transactions
            WHERE type = 'CREDIT'
            AND status = 'PENDING'
            AND scheduled_date <= CURRENT_DATE + 30
        """;
        
        BigDecimal inflows = jdbcTemplate.queryForObject(inflowSQL, BigDecimal.class);
        
        return outflows.subtract(inflows);
    }
    
    private BigDecimal getCurrentCashPosition() {
        String sql = """
            SELECT COALESCE(SUM(balance), 0)
            FROM accounts
            WHERE account_type IN ('CASH', 'SETTLEMENT')
        """;
        
        return jdbcTemplate.queryForObject(sql, BigDecimal.class);
    }
    
    private void monitorIntradayLiquidity(LiquidityMetrics metrics) {
        // Check if current position is below threshold
        BigDecimal minimumLiquidity = new BigDecimal("1000000"); // $1M minimum
        
        if (metrics.currentCash.compareTo(minimumLiquidity) < 0) {
            log.warn("LIQUIDITY WARNING: Current cash {} below minimum {}", 
                     metrics.currentCash, minimumLiquidity);
            
            // Trigger liquidity contingency
            triggerLiquidityContingency(metrics);
        }
        
        // Check liquidity usage
        BigDecimal usage = calculateIntradayLiquidityUsage();
        if (usage.compareTo(new BigDecimal("0.8")) > 0) {
            log.warn("LIQUIDITY WARNING: Intraday usage at {}%", 
                     usage.multiply(new BigDecimal("100")));
        }
    }
    
    private void checkLCR(BigDecimal lcr) {
        if (lcr.compareTo(LCR_MINIMUM) < 0) {
            log.error("REGULATORY BREACH: LCR {} below minimum {}", lcr, LCR_MINIMUM);
            // Trigger regulatory reporting
        }
    }
    
    private void checkNSFR(BigDecimal nsfr) {
        if (nsfr.compareTo(NSFR_MINIMUM) < 0) {
            log.error("REGULATORY BREACH: NSFR {} below minimum {}", nsfr, NSFR_MINIMUM);
            // Trigger regulatory reporting
        }
    }
    
    private void storeLiquidityMetrics(LiquidityMetrics metrics) {
        String sql = """
            INSERT INTO liquidity_metrics (
                timestamp, lcr, nsfr, current_cash, available_credit, stressed_outflows
            ) VALUES (?, ?, ?, ?, ?, ?)
        """;
        
        jdbcTemplate.update(sql,
            LocalDateTime.now(),
            metrics.lcr,
            metrics.nsfr,
            metrics.currentCash,
            metrics.availableCredit,
            metrics.stressedOutflows
        );
    }
    
    // Helper methods
    private BigDecimal calculateAvailableStableFunding() {
        return getCurrentCashPosition().multiply(new BigDecimal("1.1"));
    }
    
    private BigDecimal calculateRequiredStableFunding() {
        return getCurrentCashPosition();
    }
    
    private BigDecimal getAvailableCreditLines() {
        return new BigDecimal("5000000"); // $5M credit line
    }
    
    private BigDecimal calculateStressedOutflows() {
        return calculateNetCashOutflows().multiply(new BigDecimal("1.5"));
    }
    
    private BigDecimal calculateIntradayLiquidityUsage() {
        return new BigDecimal("0.65"); // 65% usage
    }
    
    private void triggerLiquidityContingency(LiquidityMetrics metrics) {
        log.error("LIQUIDITY CONTINGENCY TRIGGERED");
        // Implement contingency funding plan
    }
    
    public static class LiquidityMetrics {
        public BigDecimal lcr;           // Liquidity Coverage Ratio
        public BigDecimal nsfr;          // Net Stable Funding Ratio
        public BigDecimal currentCash;
        public BigDecimal availableCredit;
        public BigDecimal stressedOutflows;
        public LocalDateTime timestamp = LocalDateTime.now();
    }
}

