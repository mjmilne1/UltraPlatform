package com.ultraplatform.banking.service;

import com.ultraplatform.banking.dto.*;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.github.resilience4j.ratelimiter.RateLimiter;
import io.github.resilience4j.ratelimiter.RequestNotPermitted;
import io.github.resilience4j.retry.Retry;
import io.github.resilience4j.bulkhead.Bulkhead;
import io.github.resilience4j.bulkhead.BulkheadFullException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.function.Supplier;

@Service
@RequiredArgsConstructor
@Slf4j
public class ResilientTransactionService {
    
    private final TransactionService transactionService;
    private final CircuitBreaker transactionCircuitBreaker;
    private final RateLimiter transactionRateLimiter;
    private final RateLimiter premiumRateLimiter;
    private final Retry transactionRetry;
    private final Bulkhead transactionBulkhead;
    
    public TransferResponse transferFundsWithResilience(TransferRequest request) {
        // Determine rate limiter based on account type (example logic)
        RateLimiter rateLimiter = isPremiumAccount(request.getFromAccountId()) 
            ? premiumRateLimiter 
            : transactionRateLimiter;
        
        // Build resilient call chain
        Supplier<TransferResponse> decoratedSupplier = () -> 
            transactionService.transferFunds(request);
        
        // Apply resilience patterns in order:
        // 1. Rate Limiting (prevent overload)
        decoratedSupplier = RateLimiter.decorateSupplier(rateLimiter, decoratedSupplier);
        
        // 2. Bulkhead (isolate concurrent calls)
        decoratedSupplier = Bulkhead.decorateSupplier(transactionBulkhead, decoratedSupplier);
        
        // 3. Circuit Breaker (fail fast on system issues)
        decoratedSupplier = CircuitBreaker.decorateSupplier(transactionCircuitBreaker, decoratedSupplier);
        
        // 4. Retry (handle transient failures)
        decoratedSupplier = Retry.decorateSupplier(transactionRetry, decoratedSupplier);
        
        try {
            TransferResponse response = decoratedSupplier.get();
            log.info("Transfer completed successfully with resilience: {}", 
                    response.getTransactionRef());
            return response;
            
        } catch (RequestNotPermitted e) {
            log.error("Rate limit exceeded for transfer request");
            return createFallbackResponse(request, "RATE_LIMITED", 
                    "Too many requests. Please try again later.");
                    
        } catch (BulkheadFullException e) {
            log.error("System at maximum capacity");
            return createFallbackResponse(request, "CAPACITY_EXCEEDED", 
                    "System is busy. Please try again in a moment.");
                    
        } catch (Exception e) {
            log.error("Transfer failed after resilience attempts: {}", e.getMessage());
            return createFallbackResponse(request, "FAILED", 
                    "Transfer could not be completed. Please contact support.");
        }
    }
    
    // Fallback method when all resilience patterns fail
    private TransferResponse createFallbackResponse(TransferRequest request, 
                                                   String status, 
                                                   String message) {
        return TransferResponse.builder()
            .transactionRef("FALLBACK-" + UUID.randomUUID())
            .amount(request.getAmount())
            .fromAccountId(request.getFromAccountId())
            .toAccountId(request.getToAccountId())
            .status(status)
            .message(message)
            .timestamp(Instant.now())
            .build();
    }
    
    private boolean isPremiumAccount(UUID accountId) {
        // Logic to determine if account is premium
        // For demo, randomly assign premium status
        return accountId.toString().hashCode() % 2 == 0;
    }
    
    // Health check method with circuit breaker
    public String getSystemHealth() {
        return transactionCircuitBreaker.decorateSupplier(() -> {
            // Check various system components
            String kafkaHealth = checkKafkaHealth();
            String dbHealth = checkDatabaseHealth();
            
            return String.format("System Health - Circuit: %s, Kafka: %s, DB: %s",
                transactionCircuitBreaker.getState(),
                kafkaHealth,
                dbHealth);
        }).get();
    }
    
    private String checkKafkaHealth() {
        // Implement Kafka health check
        return "UP";
    }
    
    private String checkDatabaseHealth() {
        // Implement database health check
        return "UP";
    }
    
    // Get resilience metrics
    public ResilienceMetrics getResilienceMetrics() {
        return ResilienceMetrics.builder()
            .circuitBreakerState(transactionCircuitBreaker.getState().toString())
            .circuitBreakerFailureRate(transactionCircuitBreaker.getMetrics().getFailureRate())
            .circuitBreakerSlowCallRate(transactionCircuitBreaker.getMetrics().getSlowCallRate())
            .rateLimiterAvailablePermissions(transactionRateLimiter.getMetrics().getAvailablePermissions())
            .bulkheadAvailableConcurrentCalls(transactionBulkhead.getMetrics().getAvailableConcurrentCalls())
            .retryMetrics(transactionRetry.getMetrics().getNumberOfSuccessfulCallsWithRetryAttempt())
            .build();
    }
}

