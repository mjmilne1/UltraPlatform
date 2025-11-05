package com.ultraplatform.banking.service;

import com.ultraplatform.banking.entity.LedgerEvent;
import com.ultraplatform.banking.entity.Transaction;
import com.ultraplatform.banking.entity.Account;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.stereotype.Service;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

@Service
@RequiredArgsConstructor
@Slf4j
public class KafkaEventPublisher {
    
    private final KafkaTemplate<String, Object> kafkaTemplate;
    
    // Publish account events
    public void publishAccountEvent(Account account, String eventType) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventId", UUID.randomUUID().toString());
        event.put("eventType", eventType);
        event.put("accountId", account.getId());
        event.put("accountNumber", account.getAccountNumber());
        event.put("balance", account.getBalance());
        event.put("currency", account.getCurrency());
        event.put("timestamp", System.currentTimeMillis());
        
        CompletableFuture<SendResult<String, Object>> future = 
            kafkaTemplate.send("account-events", account.getId().toString(), event);
        
        future.whenComplete((result, ex) -> {
            if (ex != null) {
                log.error("Failed to publish account event: {}", ex.getMessage());
            } else {
                log.info("Published account event: {} for account {}", 
                        eventType, account.getId());
            }
        });
    }
    
    // Publish transaction events
    public void publishTransactionEvent(Transaction transaction, String eventType) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventId", UUID.randomUUID().toString());
        event.put("eventType", eventType);
        event.put("transactionId", transaction.getId());
        event.put("transactionRef", transaction.getTransactionRef());
        event.put("accountId", transaction.getAccountId());
        event.put("amount", transaction.getAmount());
        event.put("type", transaction.getType());
        event.put("status", transaction.getStatus());
        event.put("timestamp", System.currentTimeMillis());
        
        kafkaTemplate.send("transaction-events", 
                          transaction.getAccountId().toString(), 
                          event);
        
        log.info("Published transaction event: {} for transaction {}", 
                eventType, transaction.getId());
    }
    
    // Publish ledger events for event sourcing
    public void publishLedgerEvent(LedgerEvent ledgerEvent) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventId", ledgerEvent.getId());
        event.put("aggregateId", ledgerEvent.getAggregateId());
        event.put("eventType", ledgerEvent.getEventType());
        event.put("sequenceNumber", ledgerEvent.getSequenceNumber());
        event.put("validFrom", ledgerEvent.getValidFrom());
        event.put("validTo", ledgerEvent.getValidTo());
        event.put("transactionTime", ledgerEvent.getTransactionTime());
        event.put("eventData", ledgerEvent.getEventData());
        
        kafkaTemplate.send("audit-events", 
                          ledgerEvent.getAggregateId().toString(), 
                          event);
        
        log.info("Published ledger event: {} for aggregate {}", 
                ledgerEvent.getEventType(), ledgerEvent.getAggregateId());
    }
}

