package com.ultraplatform.banking.risk;

import com.ultraplatform.banking.agent.AutonomousAgent;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Component
@Slf4j
public class LiquidityLearningAgent extends AutonomousAgent {
    
    @Autowired
    private JdbcTemplate jdbcTemplate;
    
    private Map<Integer, HourlyPattern> hourlyPatterns = new ConcurrentHashMap<>();
    private Map<DayOfWeek, DailyPattern> dailyPatterns = new ConcurrentHashMap<>();
    private BigDecimal predictedNeed = BigDecimal.ZERO;
    private BigDecimal optimalBuffer = new BigDecimal("1000000");
    
    @Override
    protected void performStartup() {
        log.info("Liquidity Learning Agent starting");
        // Skip loading for now to avoid SQL errors
        // loadHistoricalPatterns();
    }
    
    @Override
    protected Analysis analyze(Context context) {
        LiquidityState state = (LiquidityState) context.data;
        Analysis analysis = new Analysis();
        
        BigDecimal predicted = predictLiquidityNeed(LocalDateTime.now());
        BigDecimal actual = state.currentPosition;
        
        double stressIndicator = calculateStressIndicator(actual, predicted, state);
        
        if (stressIndicator > 0.8) {
            analysis.recommendation = "LIQUIDITY_ALERT";
            analysis.riskScore = 0.9;
            analysis.features.put("predicted_shortfall", predicted.subtract(actual).doubleValue());
        } else if (stressIndicator > 0.6) {
            analysis.recommendation = "MONITOR_CLOSELY";
            analysis.riskScore = 0.6;
        } else {
            analysis.recommendation = "NORMAL";
            analysis.riskScore = stressIndicator;
            updatePatterns(state);
        }
        
        return analysis;
    }
    
    @Override
    protected String determineAction(Analysis analysis) {
        return analysis.recommendation;
    }
    
    @Override
    protected double calculateConfidence(Analysis analysis) {
        int dataPoints = hourlyPatterns.size() * Math.max(1, dailyPatterns.size());
        return Math.min(0.95, 0.5 + (dataPoints * 0.01));
    }
    
    @Override
    protected void execute(Decision decision) {
        log.info("Liquidity decision: {}", decision.action);
    }
    
    @Override
    protected void processMessage(AgentMessage message) {
        // Handle messages
    }
    
    @Override
    protected void performHealing() {
        log.info("Recalibrating liquidity patterns");
    }
    
    @Scheduled(fixedDelay = 60000)
    public void monitorAndPredict() {
        if (state != AgentState.ACTIVE) return;
        
        LiquidityState currentState = getCurrentLiquidityState();
        Context context = new Context();
        context.data = currentState;
        Decision decision = makeDecision(context).join();
        
        if ("LIQUIDITY_ALERT".equals(decision.action)) {
            log.warn("Liquidity alert triggered");
        }
    }
    
    private BigDecimal predictLiquidityNeed(LocalDateTime when) {
        int hour = when.getHour();
        DayOfWeek day = when.getDayOfWeek();
        
        HourlyPattern hourly = hourlyPatterns.get(hour);
        DailyPattern daily = dailyPatterns.get(day);
        
        BigDecimal baseNeed = new BigDecimal("5000000");
        
        if (hourly != null) {
            baseNeed = baseNeed.multiply(hourly.multiplier);
        }
        
        if (daily != null) {
            baseNeed = baseNeed.multiply(daily.multiplier);
        }
        
        return baseNeed;
    }
    
    private double calculateStressIndicator(BigDecimal actual, BigDecimal predicted, LiquidityState state) {
        if (predicted.compareTo(BigDecimal.ZERO) == 0) return 0.0;
        double positionStress = 1.0 - (actual.doubleValue() / predicted.doubleValue());
        double velocityStress = state.inflowVelocity > 0 ? state.outflowVelocity / state.inflowVelocity : 1.0;
        return (positionStress * 0.5) + (velocityStress * 0.5);
    }
    
    private void updatePatterns(LiquidityState state) {
        int hour = LocalDateTime.now().getHour();
        DayOfWeek day = LocalDateTime.now().getDayOfWeek();
        
        HourlyPattern hourly = hourlyPatterns.computeIfAbsent(hour, h -> new HourlyPattern());
        hourly.update(state.currentPosition);
        
        DailyPattern daily = dailyPatterns.computeIfAbsent(day, d -> new DailyPattern());
        daily.update(state.currentPosition);
    }
    
    private void loadHistoricalPatterns() {
        // Disabled for now - would load from database when available
        log.info("Historical pattern loading disabled");
    }
    
    private LiquidityState getCurrentLiquidityState() {
        LiquidityState state = new LiquidityState();
        state.currentPosition = new BigDecimal("5000000");
        state.outflowVelocity = 100.0;
        state.inflowVelocity = 120.0;
        return state;
    }
    
    static class LiquidityState {
        BigDecimal currentPosition = BigDecimal.ZERO;
        double outflowVelocity = 0.0;
        double inflowVelocity = 0.0;
    }
    
    static class HourlyPattern {
        BigDecimal averagePosition = BigDecimal.ZERO;
        BigDecimal multiplier = BigDecimal.ONE;
        
        void update(BigDecimal position) {
            BigDecimal alpha = new BigDecimal("0.1");
            averagePosition = averagePosition.multiply(BigDecimal.ONE.subtract(alpha))
                            .add(position.multiply(alpha));
        }
    }
    
    static class DailyPattern {
        BigDecimal averagePosition = BigDecimal.ZERO;
        BigDecimal multiplier = BigDecimal.ONE;
        
        void update(BigDecimal position) {
            BigDecimal alpha = new BigDecimal("0.05");
            averagePosition = averagePosition.multiply(BigDecimal.ONE.subtract(alpha))
                            .add(position.multiply(alpha));
        }
    }
}

