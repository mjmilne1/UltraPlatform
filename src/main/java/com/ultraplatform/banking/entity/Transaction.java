package com.ultraplatform.banking.entity;

import jakarta.persistence.*;
import lombok.*;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "transactions")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Transaction {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;
    
    @Column(unique = true, nullable = false)
    private String transactionRef;
    
    @Column(nullable = false)
    private UUID accountId;
    
    @Column(nullable = false)
    private String type;
    
    @Column(nullable = false)
    private String entryType;
    
    @Column(nullable = false, precision = 19, scale = 4)
    private BigDecimal amount;
    
    @Column(nullable = false, precision = 19, scale = 4)
    private BigDecimal balanceAfter;
    
    @Column(nullable = false)
    private String status = "PENDING";
    
    @Column
    private String description;
    
    @Column(nullable = false)
    private Instant createdAt = Instant.now();
    
    @Version
    private Long version;
}

