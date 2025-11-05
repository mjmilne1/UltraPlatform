package com.ultraplatform.banking.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.support.Acknowledgment;
import org.springframework.kafka.support.KafkaHeaders;
import org.springframework.messaging.handler.annotation.Header;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.stereotype.Service;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class KafkaEventConsumer {
    
    private final EventSourcingService eventSourcingService;
    
    @KafkaListener(topics = "transaction-events", 
                   groupId = "ultraledger-consumer-group",
                   containerFactory = "kafkaListenerContainerFactory")
    public void consumeTransactionEvent(@Payload Map<String, Object> event,
                                       @Header(KafkaHeaders.RECEIVED_TOPIC) String topic,
                                       @Header(KafkaHeaders.RECEIVED_PARTITION) int partition,
                                       @Header(KafkaHeaders.OFFSET) long offset,
                                       Acknowledgment acknowledgment) {
        try {
            log.info("Consumed transaction event from topic {} partition {} offset {}: {}", 
                    topic, partition, offset, event.get("eventType"));
            
            // Process the event
            processTransactionEvent(event);
            
            // Acknowledge message after successful processing
            acknowledgment.acknowledge();
            
        } catch (Exception e) {
            log.error("Error processing transaction event: {}", e.getMessage());
            // Don't acknowledge - message will be redelivered
        }
    }
    
    @KafkaListener(topics = "account-events", 
                   groupId = "ultraledger-consumer-group")
    public void consumeAccountEvent(@Payload Map<String, Object> event,
                                   Acknowledgment acknowledgment) {
        try {
            log.info("Consumed account event: {}", event.get("eventType"));
            
            // Process account event
            processAccountEvent(event);
            
            acknowledgment.acknowledge();
            
        } catch (Exception e) {
            log.error("Error processing account event: {}", e.getMessage());
        }
    }
    
    @KafkaListener(topics = "audit-events", 
                   groupId = "audit-consumer-group")
    public void consumeAuditEvent(@Payload Map<String, Object> event,
                                 Acknowledgment acknowledgment) {
        try {
            log.info("Consumed audit event for aggregate: {}", 
                    event.get("aggregateId"));
            
            // Store for audit trail
            storeAuditEvent(event);
            
            acknowledgment.acknowledge();
            
        } catch (Exception e) {
            log.error("Error processing audit event: {}", e.getMessage());
        }
    }
    
    private void processTransactionEvent(Map<String, Object> event) {
        // Process transaction events
        String eventType = (String) event.get("eventType");
        
        switch (eventType) {
            case "TRANSFER_INITIATED":
                log.info("Processing transfer initiation");
                break;
            case "TRANSFER_COMPLETED":
                log.info("Processing transfer completion");
                break;
            case "TRANSFER_FAILED":
                log.info("Processing transfer failure");
                break;
        }
    }
    
    private void processAccountEvent(Map<String, Object> event) {
        // Process account events
        String eventType = (String) event.get("eventType");
        log.info("Processing account event: {}", eventType);
    }
    
    private void storeAuditEvent(Map<String, Object> event) {
        // Store audit event for compliance
        log.info("Storing audit event: {}", event.get("eventId"));
    }
}

