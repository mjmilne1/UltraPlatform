package com.ultra.platform.entity;

import lombok.*;
import java.math.BigDecimal;
import java.util.*;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Portfolio {
    private UUID id;
    private String clientName;
    private UUID ledgerAccountId; // Links to UltraLedger2
    private BigDecimal cashBalance;
    private BigDecimal totalValue;
    private List<Position> positions;
    private Date createdAt;
    private Date lastUpdated;
}
