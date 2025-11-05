package com.ultraplatform.banking.agent;

import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
@Slf4j
public class HealthMonitorAgent extends AutonomousAgent {
    
    private long lastCheckTime = System.currentTimeMillis();
    private double systemLoad = 0.0;
    
    @Override
    protected void performStartup() {
        confidence = 0.8;
        log.info("Health Monitor Agent started");
    }
    
    @Override
    protected Analysis analyze(Context context) {
        Analysis analysis = new Analysis();
        
        Runtime runtime = Runtime.getRuntime();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        double memoryUsage = (double)(totalMemory - freeMemory) / totalMemory;
        
        analysis.features.put("memory_usage", memoryUsage);
        analysis.features.put("cpu_cores", (double)runtime.availableProcessors());
        analysis.riskScore = memoryUsage > 0.8 ? memoryUsage : 0.0;
        
        if (memoryUsage > 0.9) {
            analysis.recommendation = "CRITICAL_HEAL";
        } else if (memoryUsage > 0.7) {
            analysis.recommendation = "OPTIMIZE";
        } else {
            analysis.recommendation = "HEALTHY";
        }
        
        return analysis;
    }
    
    @Override
    protected String determineAction(Analysis analysis) {
        return analysis.recommendation;
    }
    
    @Override
    protected double calculateConfidence(Analysis analysis) {
        return 0.9;
    }
    
    @Override
    protected void execute(Decision decision) {
        switch (decision.action) {
            case "CRITICAL_HEAL":
                log.warn("Executing critical healing - forcing garbage collection");
                System.gc();
                break;
            case "OPTIMIZE":
                log.info("Optimizing system resources");
                break;
            case "HEALTHY":
                log.debug("System healthy");
                break;
        }
    }
    
    @Override
    protected void processMessage(AgentMessage message) {
        if (message.type == AgentMessage.Type.HEALTH_CHECK) {
            performHealthCheck();
        }
    }
    
    @Override
    protected void performHealing() {
        log.info("Health Monitor performing self-recovery");
        System.gc();
    }
    
    @Scheduled(fixedDelay = 15000)
    public void monitorSystem() {
        if (state == AgentState.ACTIVE) {
            Context context = new Context();
            context.data = "system_check";
            makeDecision(context);
        }
    }
    
    private void performHealthCheck() {
        Runtime runtime = Runtime.getRuntime();
        long used = runtime.totalMemory() - runtime.freeMemory();
        log.info("Health Check - Memory: {}MB used", used / 1024 / 1024);
    }
}

