package com.ultraplatform.banking.service;

import com.ultraplatform.banking.entity.LedgerEvent;
import com.ultraplatform.banking.repository.LedgerEventRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.time.Instant;
import java.util.*;
import java.math.BigDecimal;

@Service
@RequiredArgsConstructor
@Slf4j
public class EventSourcingService {
    
    private final LedgerEventRepository eventRepository;
    
    @Transactional
    public void saveEvent(UUID aggregateId, String eventType, 
                         String eventData, Instant validFrom) {
        
        Long nextSequence = eventRepository
            .findTopByAggregateIdOrderBySequenceNumberDesc(aggregateId)
            .map(e -> e.getSequenceNumber() + 1)
            .orElse(1L);
        
        LedgerEvent event = LedgerEvent.builder()
            .aggregateId(aggregateId)
            .aggregateType("Account")
            .sequenceNumber(nextSequence)
            .eventType(eventType)
            .transactionTime(Instant.now())
            .validFrom(validFrom)
            .eventData(eventData)
            .correlationId(UUID.randomUUID())
            .userId("system")
            .build();
        
        eventRepository.save(event);
        log.info("Saved event: {} for aggregate: {}", eventType, aggregateId);
    }
    
    public BigDecimal getBalanceAsOf(UUID accountId, Instant asOf) {
        List<LedgerEvent> events = eventRepository
            .findByAggregateIdAndValidFromLessThanEqualOrderBySequenceNumber(
                accountId, asOf);
        
        BigDecimal balance = BigDecimal.ZERO;
        for (LedgerEvent event : events) {
            if (event.getValidTo() != null && event.getValidTo().isBefore(asOf)) {
                continue;
            }
            balance = applyEventToBalance(balance, event);
        }
        return balance;
    }
    
    public BigDecimal getBitemporalBalance(UUID accountId, 
                                          Instant validTime, 
                                          Instant transactionTime) {
        List<LedgerEvent> events = eventRepository
            .findBitemporalEvents(accountId, validTime, transactionTime);
        
        BigDecimal balance = BigDecimal.ZERO;
        for (LedgerEvent event : events) {
            balance = applyEventToBalance(balance, event);
        }
        return balance;
    }
    
    private BigDecimal applyEventToBalance(BigDecimal balance, LedgerEvent event) {
        if (event.getEventType().contains("CREDIT")) {
            return balance.add(new BigDecimal("100.00"));
        } else if (event.getEventType().contains("DEBIT")) {
            return balance.subtract(new BigDecimal("100.00"));
        }
        return balance;
    }
}

