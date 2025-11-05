package com.ultraplatform.banking.controller;

import com.ultraplatform.banking.dto.*;
import com.ultraplatform.banking.service.*;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/resilient")
@RequiredArgsConstructor
@Tag(name = "Resilient Operations", description = "Production-hardened endpoints with resilience patterns")
public class ResilientTransactionController {
    
    private final ResilientTransactionService resilientTransactionService;
    
    @PostMapping("/transfer")
    @Operation(summary = "Transfer funds with resilience", 
               description = "Transfer with circuit breaker, retry, rate limiting, and bulkhead patterns")
    public ResponseEntity<TransferResponse> transferWithResilience(@RequestBody TransferRequest request) {
        TransferResponse response = resilientTransactionService.transferFundsWithResilience(request);
        
        if ("SUCCESS".equals(response.getStatus())) {
            return ResponseEntity.ok(response);
        } else if ("RATE_LIMITED".equals(response.getStatus())) {
            return ResponseEntity.status(429).body(response);
        } else if ("CAPACITY_EXCEEDED".equals(response.getStatus())) {
            return ResponseEntity.status(503).body(response);
        } else {
            return ResponseEntity.status(500).body(response);
        }
    }
    
    @GetMapping("/health")
    @Operation(summary = "Get system health with circuit breaker status")
    public ResponseEntity<String> getHealth() {
        return ResponseEntity.ok(resilientTransactionService.getSystemHealth());
    }
    
    @GetMapping("/metrics")
    @Operation(summary = "Get resilience metrics", 
               description = "View circuit breaker, rate limiter, and bulkhead metrics")
    public ResponseEntity<ResilienceMetrics> getMetrics() {
        return ResponseEntity.ok(resilientTransactionService.getResilienceMetrics());
    }
}

