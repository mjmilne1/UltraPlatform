package com.ultraplatform.banking.risk;

import com.ultraplatform.banking.agent.AutonomousAgent;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import java.math.BigDecimal;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Component
@Slf4j
public class RiskLearningAgent extends AutonomousAgent {
    
    @Autowired
    private JdbcTemplate jdbcTemplate;
    
    // Learns normal vs abnormal patterns per account
    private Map<UUID, RiskProfile> learnedProfiles = new ConcurrentHashMap<>();
    
    @Override
    protected void performStartup() {
        log.info("Risk Learning Agent starting - loading historical profiles");
        loadHistoricalProfiles();
    }
    
    @Override
    protected Analysis analyze(Context context) {
        RiskData data = (RiskData) context.data;
        RiskProfile profile = learnedProfiles.computeIfAbsent(
            data.accountId, 
            id -> new RiskProfile(id)
        );
        
        // Compare current behavior to learned profile
        double deviation = calculateDeviation(data, profile);
        
        Analysis analysis = new Analysis();
        if (deviation > profile.getThreshold()) {
            analysis.riskScore = 0.9;
            analysis.recommendation = "INVESTIGATE";
            analysis.features.put("deviation", deviation);
            analysis.features.put("threshold", profile.getThreshold());
            log.warn("Account {} showing abnormal behavior: deviation={}", 
                     data.accountId, deviation);
        } else {
            // Update profile with new normal behavior
            profile.update(data);
            analysis.riskScore = deviation / profile.getThreshold();
            analysis.recommendation = "NORMAL";
        }
        
        return analysis;
    }
    
    @Override
    protected String determineAction(Analysis analysis) {
        return analysis.recommendation;
    }
    
    @Override
    protected double calculateConfidence(Analysis analysis) {
        // Higher confidence when deviation is very high or very low
        double deviation = analysis.features.getOrDefault("deviation", 0.5);
        if (deviation > 2.0 || deviation < 0.2) {
            return 0.95;
        }
        return 0.7;
    }
    
    @Override
    protected void execute(Decision decision) {
        if ("INVESTIGATE".equals(decision.action)) {
            // Trigger risk investigation workflow
            log.warn("Triggering risk investigation: {}", decision.parameters);
        }
    }
    
    @Override
    protected void processMessage(AgentMessage message) {
        // Handle messages from other agents
    }
    
    @Override
    protected void performHealing() {
        // Reset profiles if accuracy drops
        log.info("Performing self-healing - recalibrating risk profiles");
        recalibrateProfiles();
    }
    
    @Scheduled(fixedDelay = 60000) // Every minute
    public void monitorRisk() {
        if (state != AgentState.ACTIVE) return;
        
        // Get recent transactions
        List<RiskData> recentData = getRecentRiskData();
        
        for (RiskData data : recentData) {
            Context context = new Context();
            context.data = data;
            makeDecision(context);
        }
    }
    
    private double calculateDeviation(RiskData current, RiskProfile profile) {
        double deviation = 0.0;
        
        // Amount deviation
        double amountDev = Math.abs(current.amount.doubleValue() - profile.avgAmount) 
                          / profile.stdAmount;
        
        // Frequency deviation
        double freqDev = Math.abs(current.frequency - profile.avgFrequency) 
                        / Math.max(profile.stdFrequency, 1.0);
        
        // Time pattern deviation
        double timeDev = calculateTimeDeviation(current.hour, profile);
        
        // Weighted combination
        deviation = (amountDev * 0.4) + (freqDev * 0.3) + (timeDev * 0.3);
        
        return deviation;
    }
    
    private double calculateTimeDeviation(int hour, RiskProfile profile) {
        Double expectedActivity = profile.hourlyPattern.get(hour);
        if (expectedActivity == null) {
            return 1.0; // New time slot
        }
        return Math.abs(1.0 - expectedActivity);
    }
    
    private void loadHistoricalProfiles() {
        String sql = """
            SELECT 
                account_id,
                AVG(amount) as avg_amount,
                STDDEV(amount) as std_amount,
                COUNT(*) / COUNT(DISTINCT DATE(created_at)) as avg_frequency
            FROM transactions
            WHERE created_at > CURRENT_DATE - 90
            GROUP BY account_id
        """;
        
        jdbcTemplate.query(sql, (rs) -> {
            UUID accountId = UUID.fromString(rs.getString("account_id"));
            RiskProfile profile = new RiskProfile(accountId);
            profile.avgAmount = rs.getDouble("avg_amount");
            profile.stdAmount = rs.getDouble("std_amount");
            profile.avgFrequency = rs.getDouble("avg_frequency");
            learnedProfiles.put(accountId, profile);
        });
        
        log.info("Loaded {} risk profiles", learnedProfiles.size());
    }
    
    private List<RiskData> getRecentRiskData() {
        String sql = """
            SELECT 
                account_id,
                amount,
                EXTRACT(HOUR FROM created_at) as hour,
                COUNT(*) OVER (PARTITION BY account_id, DATE(created_at)) as daily_count
            FROM transactions
            WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '5 minutes'
        """;
        
        return jdbcTemplate.query(sql, (rs, rowNum) -> {
            RiskData data = new RiskData();
            data.accountId = UUID.fromString(rs.getString("account_id"));
            data.amount = rs.getBigDecimal("amount");
            data.hour = rs.getInt("hour");
            data.frequency = rs.getInt("daily_count");
            return data;
        });
    }
    
    private void recalibrateProfiles() {
        // Reduce confidence in all profiles
        learnedProfiles.values().forEach(profile -> {
            profile.confidence *= 0.8;
        });
    }
    
    // Inner classes
    public static class RiskData {
        UUID accountId;
        BigDecimal amount;
        int frequency;
        int hour;
    }
    
    static class RiskProfile {
        UUID accountId;
        double avgAmount = 0.0;
        double stdAmount = 1.0;
        double avgFrequency = 0.0;
        double stdFrequency = 1.0;
        Map<Integer, Double> hourlyPattern = new HashMap<>();
        double threshold = 2.5; // 2.5 standard deviations
        double confidence = 0.5;
        
        RiskProfile(UUID accountId) {
            this.accountId = accountId;
        }
        
        void update(RiskData data) {
            // Exponential moving average update
            double alpha = 0.1;
            avgAmount = (1 - alpha) * avgAmount + alpha * data.amount.doubleValue();
            avgFrequency = (1 - alpha) * avgFrequency + alpha * data.frequency;
            
            // Update hourly pattern
            hourlyPattern.merge(data.hour, 1.0, (old, one) -> (old * 0.9) + 0.1);
            
            // Increase confidence as we learn
            confidence = Math.min(0.95, confidence + 0.01);
        }
        
        double getThreshold() {
            // Dynamic threshold based on confidence
            return threshold * (2.0 - confidence);
        }
    }
}

