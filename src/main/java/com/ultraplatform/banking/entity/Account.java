package com.ultraplatform.banking.entity;

import jakarta.persistence.*;
import lombok.*;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "accounts")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Account {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;
    
    @Column(unique = true, nullable = false)
    private String accountNumber;
    
    @Column(nullable = false)
    private String customerName;
    
    @Column(nullable = false, precision = 19, scale = 4)
    private BigDecimal balance = BigDecimal.ZERO;
    
    @Column(nullable = false, precision = 19, scale = 4)
    private BigDecimal availableBalance = BigDecimal.ZERO;
    
    @Column(nullable = false)
    private String currency = "USD";
    
    @Column(nullable = false)
    private String status = "ACTIVE";
    
    @Column(nullable = false)
    private Instant createdAt = Instant.now();
    
    @Version
    private Long version;
}

