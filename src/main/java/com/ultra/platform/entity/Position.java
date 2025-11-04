package com.ultra.platform.entity;

import lombok.*;
import java.math.BigDecimal;
import java.util.Date;
import java.util.UUID;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Position {
    private UUID id;
    private UUID portfolioId;
    private String symbol;
    private int quantity;
    private BigDecimal averageCost;
    private BigDecimal currentPrice;
    private BigDecimal marketValue;
    private BigDecimal unrealizedPnL;
    private Date openedAt;
}
