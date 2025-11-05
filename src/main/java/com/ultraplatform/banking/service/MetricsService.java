package com.ultraplatform.banking.service;

import io.micrometer.core.instrument.*;
import io.micrometer.core.instrument.Timer;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

@Service
@Slf4j
public class MetricsService {
    
    private final MeterRegistry meterRegistry;
    private final Counter transactionCounter;
    private final Counter successfulTransactionCounter;
    private final Counter failedTransactionCounter;
    private final Counter kafkaEventCounter;
    private final Gauge activeAccountsGauge;
    private final Timer transactionTimer;
    private final DistributionSummary transactionAmountSummary;
    private final AtomicLong totalVolume;
    private final AtomicInteger activeAccounts;
    
    public MetricsService(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
        this.totalVolume = new AtomicLong(0);
        this.activeAccounts = new AtomicInteger(0);
        
        // Transaction counters
        this.transactionCounter = Counter.builder("ledger.transactions.total")
            .description("Total number of transactions")
            .tag("type", "all")
            .register(meterRegistry);
            
        this.successfulTransactionCounter = Counter.builder("ledger.transactions.success")
            .description("Successful transactions")
            .tag("status", "success")
            .register(meterRegistry);
            
        this.failedTransactionCounter = Counter.builder("ledger.transactions.failed")
            .description("Failed transactions")
            .tag("status", "failed")
            .register(meterRegistry);
            
        // Kafka metrics
        this.kafkaEventCounter = Counter.builder("ledger.kafka.events.published")
            .description("Kafka events published")
            .tag("topic", "all")
            .register(meterRegistry);
            
        // Active accounts gauge
        this.activeAccountsGauge = Gauge.builder("ledger.accounts.active", activeAccounts, AtomicInteger::get)
            .description("Number of active accounts")
            .register(meterRegistry);
            
        // Transaction timing
        this.transactionTimer = Timer.builder("ledger.transaction.duration")
            .description("Transaction processing duration")
            .publishPercentiles(0.5, 0.75, 0.95, 0.99)
            .publishPercentileHistogram()
            .register(meterRegistry);
            
        // Transaction amounts distribution
        this.transactionAmountSummary = DistributionSummary.builder("ledger.transaction.amount")
            .description("Transaction amount distribution")
            .publishPercentiles(0.5, 0.75, 0.95, 0.99)
            .baseUnit("USD")
            .register(meterRegistry);
            
        // Total volume gauge
        Gauge.builder("ledger.volume.total", totalVolume, AtomicLong::get)
            .description("Total transaction volume")
            .baseUnit("USD")
            .register(meterRegistry);
            
        // JVM metrics are automatically included
        // Database connection pool metrics are automatically included
        // Kafka consumer/producer metrics are automatically included
        
        log.info("Metrics service initialized with Prometheus registry");
    }
    
    // Record successful transaction
    public void recordSuccessfulTransaction(BigDecimal amount, long durationMs) {
        transactionCounter.increment();
        successfulTransactionCounter.increment();
        transactionTimer.record(Duration.ofMillis(durationMs));
        transactionAmountSummary.record(amount.doubleValue());
        totalVolume.addAndGet(amount.longValue());
        
        meterRegistry.counter("ledger.transactions.by_amount",
            "range", getAmountRange(amount)).increment();
    }
    
    // Record failed transaction
    public void recordFailedTransaction(String reason) {
        transactionCounter.increment();
        failedTransactionCounter.increment();
        
        meterRegistry.counter("ledger.transactions.failure.reason",
            "reason", reason).increment();
    }
    
    // Record Kafka event
    public void recordKafkaEvent(String topic, String eventType) {
        kafkaEventCounter.increment();
        
        meterRegistry.counter("ledger.kafka.events",
            "topic", topic,
            "type", eventType).increment();
    }
    
    // Record circuit breaker state change
    public void recordCircuitBreakerStateChange(String state) {
        meterRegistry.counter("ledger.circuit_breaker.state_changes",
            "state", state).increment();
    }
    
    // Record rate limit hit
    public void recordRateLimitHit(String limiterName) {
        meterRegistry.counter("ledger.rate_limit.hits",
            "limiter", limiterName).increment();
    }
    
    // Track account creation
    public void recordAccountCreation(String accountType) {
        activeAccounts.incrementAndGet();
        meterRegistry.counter("ledger.accounts.created",
            "type", accountType).increment();
    }
    
    // Track API endpoint calls
    public Timer.Sample startApiTimer() {
        return Timer.start(meterRegistry);
    }
    
    public void recordApiCall(Timer.Sample sample, String endpoint, String method, int status) {
        sample.stop(Timer.builder("ledger.api.requests")
            .tag("endpoint", endpoint)
            .tag("method", method)
            .tag("status", String.valueOf(status))
            .publishPercentiles(0.5, 0.95, 0.99)
            .register(meterRegistry));
    }
    
    // Business metrics
    public void recordDailyActiveUsers(int count) {
        meterRegistry.gauge("ledger.users.daily_active", count);
    }
    
    public void recordAccountBalance(String accountId, BigDecimal balance) {
        meterRegistry.gauge("ledger.account.balance",
            Tags.of("account", accountId.substring(0, 8)),
            balance.doubleValue());
    }
    
    private String getAmountRange(BigDecimal amount) {
        if (amount.compareTo(BigDecimal.valueOf(100)) < 0) return "0-100";
        else if (amount.compareTo(BigDecimal.valueOf(1000)) < 0) return "100-1000";
        else if (amount.compareTo(BigDecimal.valueOf(10000)) < 0) return "1000-10000";
        else return "10000+";
    }
}

