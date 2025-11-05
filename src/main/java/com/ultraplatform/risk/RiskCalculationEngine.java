package com.ultraplatform.banking.risk;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;

@Component
@Slf4j
public class RiskCalculationEngine {
    
    @Autowired
    private JdbcTemplate jdbcTemplate;
    
    private static final BigDecimal CONFIDENCE_95 = new BigDecimal("1.96");
    private static final BigDecimal CONFIDENCE_99 = new BigDecimal("2.58");
    
    @Scheduled(fixedDelay = 300000) // Every 5 minutes
    public void calculateRiskMetrics() {
        try {
            Map<String, BigDecimal> var = calculateValueAtRisk();
            Map<String, BigDecimal> exposure = calculateExposures();
            BigDecimal concentrationRisk = calculateConcentrationRisk();
            
            // Store risk metrics
            storeRiskMetrics(var, exposure, concentrationRisk);
            
            // Check risk limits
            checkRiskLimits(var, exposure, concentrationRisk);
            
        } catch (Exception e) {
            log.error("Risk calculation failed", e);
        }
    }
    
    public Map<String, BigDecimal> calculateValueAtRisk() {
        // Historical VaR calculation
        String sql = """
            SELECT 
                PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY daily_change) as var_95,
                PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY daily_change) as var_99
            FROM (
                SELECT 
                    DATE(transaction_date) as day,
                    SUM(CASE WHEN type = 'DEBIT' THEN -amount ELSE amount END) as daily_change
                FROM transactions
                WHERE created_at >= CURRENT_DATE - 90
                GROUP BY DATE(transaction_date)
            ) daily_changes
        """;
        
        Map<String, Object> result = jdbcTemplate.queryForMap(sql);
        
        Map<String, BigDecimal> var = new HashMap<>();
        var.put("VaR_95", (BigDecimal) result.get("var_95"));
        var.put("VaR_99", (BigDecimal) result.get("var_99"));
        
        log.info("VaR calculated - 95%: {}, 99%: {}", var.get("VaR_95"), var.get("VaR_99"));
        
        return var;
    }
    
    public Map<String, BigDecimal> calculateExposures() {
        Map<String, BigDecimal> exposures = new HashMap<>();
        
        // Credit exposure
        String creditSQL = """
            SELECT 
                SUM(balance) as total_credit_exposure
            FROM accounts 
            WHERE balance < 0
        """;
        
        BigDecimal creditExposure = jdbcTemplate.queryForObject(creditSQL, BigDecimal.class);
        exposures.put("credit", creditExposure != null ? creditExposure.abs() : BigDecimal.ZERO);
        
        // Large exposure
        String largeExposureSQL = """
            SELECT 
                COUNT(*) as large_exposures,
                SUM(balance) as total_large_exposure
            FROM accounts 
            WHERE ABS(balance) > 1000000
        """;
        
        Map<String, Object> largeExp = jdbcTemplate.queryForMap(largeExposureSQL);
        exposures.put("large_exposure_count", new BigDecimal(largeExp.get("large_exposures").toString()));
        exposures.put("large_exposure_total", (BigDecimal) largeExp.get("total_large_exposure"));
        
        return exposures;
    }
    
    public BigDecimal calculateConcentrationRisk() {
        // Herfindahl Index for concentration
        String sql = """
            WITH account_shares AS (
                SELECT 
                    account_id,
                    ABS(SUM(amount)) as total_amount
                FROM transactions
                WHERE created_at >= CURRENT_DATE - 30
                GROUP BY account_id
            ),
            total_volume AS (
                SELECT SUM(total_amount) as total FROM account_shares
            )
            SELECT 
                SUM(POWER(total_amount / total, 2)) as herfindahl_index
            FROM account_shares, total_volume
        """;
        
        BigDecimal hhi = jdbcTemplate.queryForObject(sql, BigDecimal.class);
        return hhi != null ? hhi : BigDecimal.ZERO;
    }
    
    private void storeRiskMetrics(Map<String, BigDecimal> var, 
                                   Map<String, BigDecimal> exposure, 
                                   BigDecimal concentration) {
        String sql = """
            INSERT INTO risk_metrics (
                metric_date, var_95, var_99, credit_exposure, 
                large_exposure_total, concentration_risk
            ) VALUES (CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
        """;
        
        jdbcTemplate.update(sql, 
            var.get("VaR_95"),
            var.get("VaR_99"),
            exposure.get("credit"),
            exposure.get("large_exposure_total"),
            concentration
        );
    }
    
    private void checkRiskLimits(Map<String, BigDecimal> var, 
                                  Map<String, BigDecimal> exposure,
                                  BigDecimal concentration) {
        // Check against risk limits
        BigDecimal varLimit = new BigDecimal("1000000");
        BigDecimal exposureLimit = new BigDecimal("5000000");
        BigDecimal concentrationLimit = new BigDecimal("0.2"); // HHI > 0.2 indicates concentration
        
        if (var.get("VaR_99").abs().compareTo(varLimit) > 0) {
            log.warn("RISK ALERT: VaR 99% exceeds limit: {}", var.get("VaR_99"));
            triggerRiskAlert("VAR_BREACH", var.get("VaR_99"));
        }
        
        if (exposure.get("credit").compareTo(exposureLimit) > 0) {
            log.warn("RISK ALERT: Credit exposure exceeds limit: {}", exposure.get("credit"));
            triggerRiskAlert("CREDIT_EXPOSURE_BREACH", exposure.get("credit"));
        }
        
        if (concentration.compareTo(concentrationLimit) > 0) {
            log.warn("RISK ALERT: Concentration risk high: {}", concentration);
            triggerRiskAlert("CONCENTRATION_RISK", concentration);
        }
    }
    
    private void triggerRiskAlert(String alertType, BigDecimal value) {
        // Send alert through Kafka or notification service
        log.error("RISK ALERT TRIGGERED: {} = {}", alertType, value);
    }
}

