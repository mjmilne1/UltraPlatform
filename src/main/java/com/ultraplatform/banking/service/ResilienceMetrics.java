package com.ultraplatform.banking.service;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ResilienceMetrics {
    private String circuitBreakerState;
    private float circuitBreakerFailureRate;
    private float circuitBreakerSlowCallRate;
    private int rateLimiterAvailablePermissions;
    private int bulkheadAvailableConcurrentCalls;
    private long retryMetrics;
}

