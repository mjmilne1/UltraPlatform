package com.ultraplatform.banking.config;

import io.github.resilience4j.ratelimiter.RateLimiter;
import io.github.resilience4j.ratelimiter.RateLimiterConfig;
import io.github.resilience4j.ratelimiter.RateLimiterRegistry;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import java.time.Duration;

@Configuration
public class RateLimitingConfig {
    
    @Bean
    public RateLimiterRegistry rateLimiterRegistry() {
        // Strict rate limiting for production
        RateLimiterConfig authConfig = RateLimiterConfig.custom()
            .limitRefreshPeriod(Duration.ofMinutes(1))
            .limitForPeriod(5)  // 5 login attempts per minute
            .timeoutDuration(Duration.ofSeconds(0))
            .build();
        
        RateLimiterConfig apiConfig = RateLimiterConfig.custom()
            .limitRefreshPeriod(Duration.ofSeconds(1))
            .limitForPeriod(100)  // 100 requests per second
            .timeoutDuration(Duration.ofMillis(100))
            .build();
        
        RateLimiterConfig transactionConfig = RateLimiterConfig.custom()
            .limitRefreshPeriod(Duration.ofSeconds(1))
            .limitForPeriod(10)  // 10 transactions per second
            .timeoutDuration(Duration.ofMillis(500))
            .build();
        
        RateLimiterRegistry registry = RateLimiterRegistry.of(apiConfig);
        registry.rateLimiter("auth-limiter", authConfig);
        registry.rateLimiter("api-limiter", apiConfig);
        registry.rateLimiter("transaction-limiter", transactionConfig);
        
        return registry;
    }
}

