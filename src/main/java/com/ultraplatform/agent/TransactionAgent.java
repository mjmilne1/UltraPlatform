package com.ultraplatform.banking.agent;

import com.ultraplatform.banking.entity.Transaction;
import com.ultraplatform.banking.service.TransactionService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import java.math.BigDecimal;
import java.util.Random;

@Component
@Slf4j
public class TransactionAgent extends AutonomousAgent {
    
    @Autowired(required = false)
    private TransactionService transactionService;
    
    private final Random random = new Random();
    
    @Override
    protected void performStartup() {
        learningWeights.put("amount", 0.3);
        learningWeights.put("risk", 0.4);
        learningWeights.put("history", 0.3);
        log.info("Transaction Agent initialized");
    }
    
    @Override
    protected Analysis analyze(Context context) {
        Analysis analysis = new Analysis();
        
        if (context.data instanceof Transaction) {
            Transaction tx = (Transaction) context.data;
            
            double amountRisk = calculateAmountRisk(tx.getAmount());
            analysis.riskScore = amountRisk;
            analysis.opportunityScore = 1.0 - amountRisk;
            analysis.features.put("amount_risk", amountRisk);
            analysis.features.put("pattern_match", checkPattern(tx));
            
            if (analysis.riskScore > 0.7) {
                analysis.recommendation = "BLOCK";
            } else if (analysis.riskScore > 0.4) {
                analysis.recommendation = "REVIEW";
            } else {
                analysis.recommendation = "APPROVE";
            }
        }
        
        return analysis;
    }
    
    @Override
    protected String determineAction(Analysis analysis) {
        return analysis.recommendation != null ? analysis.recommendation : "MONITOR";
    }
    
    @Override
    protected double calculateConfidence(Analysis analysis) {
        if (analysis.riskScore > 0.8 || analysis.riskScore < 0.2) {
            return 0.9;
        }
        return 0.5;
    }
    
    @Override
    protected void execute(Decision decision) {
        log.info("Transaction Agent executing: {} with confidence {}", 
                 decision.action, decision.confidence);
        
        switch (decision.action) {
            case "APPROVE":
                metrics.successRate = Math.min(1.0, metrics.successRate + 0.01);
                break;
            case "BLOCK":
                log.warn("Transaction blocked by agent");
                break;
            case "REVIEW":
                log.info("Transaction sent for review");
                break;
        }
    }
    
    @Override
    protected void processMessage(AgentMessage message) {
        if (message.type == AgentMessage.Type.COLLABORATION) {
            log.info("Collaborating with agent: {}", message.senderId);
        }
    }
    
    @Override
    protected void performHealing() {
        log.info("Transaction Agent performing self-healing");
        metrics.successRate = 0.5;
    }
    
    @Scheduled(fixedDelay = 30000)
    public void autonomousMonitoring() {
        if (state == AgentState.ACTIVE) {
            log.debug("Transaction Agent monitoring - Decisions: {}, Success Rate: {}%", 
                      metrics.decisionsProcessed, 
                      String.format("%.1f", metrics.successRate * 100));
            
            if (metrics.successRate < 0.7 && metrics.decisionsProcessed > 10) {
                adjustWeights();
            }
        }
    }
    
    private double calculateAmountRisk(BigDecimal amount) {
        if (amount == null) return 0.5;
        double value = amount.doubleValue();
        if (value > 100000) return 0.9;
        if (value > 10000) return 0.6;
        if (value > 1000) return 0.3;
        return 0.1;
    }
    
    private double checkPattern(Transaction tx) {
        return random.nextDouble();
    }
    
    private void adjustWeights() {
        log.info("Adjusting decision weights for optimization");
        learningWeights.forEach((key, value) -> {
            double adjustment = (random.nextDouble() - 0.5) * 0.1;
            learningWeights.put(key, Math.max(0.1, Math.min(0.9, value + adjustment)));
        });
    }
}

