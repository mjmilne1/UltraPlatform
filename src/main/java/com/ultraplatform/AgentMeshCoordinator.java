package com.ultraplatform.banking.agent;

import org.springframework.stereotype.Component;
import lombok.extern.slf4j.Slf4j;
import jakarta.annotation.PostConstruct;
import java.util.Map;
import java.util.LinkedHashMap;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import com.ultraplatform.banking.reconciliation.ReconciliationAgent;
import com.ultraplatform.banking.risk.RiskLearningAgent;
import com.ultraplatform.banking.risk.LiquidityLearningAgent;

@Component
@Slf4j
public class AgentMeshCoordinator {
    
    @Autowired
    private ApplicationContext context;
    
    private Map<String, AutonomousAgent> agents = new LinkedHashMap<>();
    
    @PostConstruct
    public void initializeMesh() {
        log.info("Starting Full Agent Mesh Coordinator - Loading all 7 agents");
        
        try {
            // Core agents
            agents.put("transaction", context.getBean(TransactionAgent.class));
            agents.put("health", context.getBean(HealthMonitorAgent.class));
            
            // Reconciliation agent
            try {
                agents.put("reconciliation", context.getBean(ReconciliationAgent.class));
            } catch (Exception e) {
                log.warn("ReconciliationAgent not loaded: {}", e.getMessage());
            }
            
            // Risk agents
            try {
                agents.put("risk", context.getBean(RiskLearningAgent.class));
            } catch (Exception e) {
                log.warn("RiskLearningAgent not loaded: {}", e.getMessage());
            }
            
            // Liquidity agent
            try {
                agents.put("liquidity", context.getBean(LiquidityLearningAgent.class));
            } catch (Exception e) {
                log.warn("LiquidityLearningAgent not loaded: {}", e.getMessage());
            }
            
            // Initialize all loaded agents
            agents.forEach((name, agent) -> {
                try {
                    agent.initialize();
                    log.info("? Initialized agent: {}", name);
                } catch (Exception e) {
                    log.error("Failed to initialize agent {}: {}", name, e.getMessage());
                }
            });
            
        } catch (Exception e) {
            log.error("Error loading agents: {}", e.getMessage());
        }
        
        log.info("Agent Mesh initialized with {} agents", agents.size());
        log.info("Active agents: {}", agents.keySet());
    }
    
    public void broadcastMessage(AutonomousAgent.AgentMessage message) {
        agents.values().forEach(agent -> agent.receiveMessage(message));
    }
    
    public Map<String, AutonomousAgent> getAgents() {
        return agents;
    }
}

