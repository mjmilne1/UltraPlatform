package com.ultraplatform.banking.entity;

import jakarta.persistence.*;
import lombok.*;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "ledger_events", indexes = {
    @Index(name = "idx_aggregate_id", columnList = "aggregateId"),
    @Index(name = "idx_valid_time", columnList = "validFrom,validTo"),
    @Index(name = "idx_transaction_time", columnList = "transactionTime")
})
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class LedgerEvent {
    
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;
    
    // Event Sourcing
    @Column(nullable = false)
    private UUID aggregateId;
    
    @Column(nullable = false)
    private String aggregateType;
    
    @Column(nullable = false)
    private Long sequenceNumber;
    
    @Column(nullable = false)
    private String eventType;
    
    // Bitemporal
    @Column(nullable = false)
    private Instant transactionTime; // When we recorded it
    
    @Column(nullable = false)
    private Instant validFrom; // When it actually happened
    
    @Column
    private Instant validTo; // When it stopped being true
    
    // Event Data
    @Column(columnDefinition = "text")
    private String eventData; // JSON payload
    
    @Column
    private UUID correlationId;
    
    @Column
    private UUID causationId;
    
    @Column(nullable = false)
    private String userId;
    
    @Version
    private Long version;
}

