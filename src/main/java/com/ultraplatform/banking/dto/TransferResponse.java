package com.ultraplatform.banking.dto;

import lombok.*;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TransferResponse {
    private String transactionRef;
    private BigDecimal amount;
    private UUID fromAccountId;
    private UUID toAccountId;
    private String status;
    private String message;  // Added for resilience fallback messages
    private Instant timestamp;
}

