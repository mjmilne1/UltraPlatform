package com.ultraplatform.banking.config;

import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.github.resilience4j.ratelimiter.RateLimiter;
import io.github.resilience4j.ratelimiter.RateLimiterConfig;
import io.github.resilience4j.ratelimiter.RateLimiterRegistry;
import io.github.resilience4j.retry.Retry;
import io.github.resilience4j.retry.RetryConfig;
import io.github.resilience4j.retry.RetryRegistry;
import io.github.resilience4j.bulkhead.Bulkhead;
import io.github.resilience4j.bulkhead.BulkheadConfig;
import io.github.resilience4j.bulkhead.BulkheadRegistry;
import io.github.resilience4j.timelimiter.TimeLimiter;
import io.github.resilience4j.timelimiter.TimeLimiterConfig;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import lombok.extern.slf4j.Slf4j;

import java.time.Duration;

@Configuration
@Slf4j
public class ResilienceConfig {
    
    // Circuit Breaker Configuration
    @Bean
    public CircuitBreaker transactionCircuitBreaker() {
        CircuitBreakerConfig config = CircuitBreakerConfig.custom()
            .failureRateThreshold(50) // Open circuit if 50% of calls fail
            .waitDurationInOpenState(Duration.ofSeconds(30)) // Wait 30s before half-open
            .slidingWindowSize(10) // Evaluate last 10 calls
            .permittedNumberOfCallsInHalfOpenState(3) // Allow 3 test calls
            .slowCallDurationThreshold(Duration.ofSeconds(5)) // Call is slow if > 5s
            .slowCallRateThreshold(50) // Open if 50% calls are slow
            .automaticTransitionFromOpenToHalfOpenEnabled(true)
            .build();
        
        CircuitBreakerRegistry registry = CircuitBreakerRegistry.of(config);
        CircuitBreaker circuitBreaker = registry.circuitBreaker("transaction-service");
        
        // Add event listeners
        circuitBreaker.getEventPublisher()
            .onStateTransition(event -> 
                log.warn("Circuit Breaker State Transition: {}", event))
            .onFailureRateExceeded(event -> 
                log.error("Circuit Breaker failure rate exceeded: {}", event));
        
        return circuitBreaker;
    }
    
    // Rate Limiter Configuration
    @Bean
    public RateLimiter transactionRateLimiter() {
        RateLimiterConfig config = RateLimiterConfig.custom()
            .limitForPeriod(100) // 100 requests
            .limitRefreshPeriod(Duration.ofSeconds(1)) // per second
            .timeoutDuration(Duration.ofSeconds(5)) // Wait max 5s for permission
            .build();
        
        RateLimiterRegistry registry = RateLimiterRegistry.of(config);
        RateLimiter rateLimiter = registry.rateLimiter("transaction-limiter");
        
        rateLimiter.getEventPublisher()
            .onSuccess(event -> log.debug("Rate limiter permission granted"))
            .onFailure(event -> log.warn("Rate limiter permission denied"));
        
        return rateLimiter;
    }
    
    // Premium Rate Limiter for VIP accounts
    @Bean
    public RateLimiter premiumRateLimiter() {
        RateLimiterConfig config = RateLimiterConfig.custom()
            .limitForPeriod(1000) // 1000 requests for premium
            .limitRefreshPeriod(Duration.ofSeconds(1))
            .timeoutDuration(Duration.ofMillis(100))
            .build();
        
        return RateLimiter.of("premium-limiter", config);
    }
    
    // Retry Configuration
    @Bean
    public Retry transactionRetry() {
        RetryConfig config = RetryConfig.custom()
            .maxAttempts(3)
            .waitDuration(Duration.ofMillis(500))
            .retryOnException(e -> !(e instanceof IllegalArgumentException))
            .retryExceptions(Exception.class)
            .ignoreExceptions(IllegalArgumentException.class)
            .build();
        
        RetryRegistry registry = RetryRegistry.of(config);
        Retry retry = registry.retry("transaction-retry");
        
        retry.getEventPublisher()
            .onRetry(event -> log.info("Retry attempt: {}", event.getNumberOfRetryAttempts()));
        
        return retry;
    }
    
    // Bulkhead Configuration (Concurrency Limiter)
    @Bean
    public Bulkhead transactionBulkhead() {
        BulkheadConfig config = BulkheadConfig.custom()
            .maxConcurrentCalls(25) // Max 25 concurrent transactions
            .maxWaitDuration(Duration.ofMillis(500))
            .build();
        
        BulkheadRegistry registry = BulkheadRegistry.of(config);
        Bulkhead bulkhead = registry.bulkhead("transaction-bulkhead");
        
        bulkhead.getEventPublisher()
            .onCallPermitted(event -> log.debug("Bulkhead call permitted"))
            .onCallRejected(event -> log.warn("Bulkhead call rejected - too many concurrent calls"));
        
        return bulkhead;
    }
    
    // Time Limiter Configuration
    @Bean
    public TimeLimiter transactionTimeLimiter() {
        TimeLimiterConfig config = TimeLimiterConfig.custom()
            .timeoutDuration(Duration.ofSeconds(3))
            .cancelRunningFuture(true)
            .build();
        
        return TimeLimiter.of("transaction-limiter", config);
    }
}

