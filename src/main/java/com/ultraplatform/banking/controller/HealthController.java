package com.ultraplatform.banking.controller;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/")
public class HealthController {
    
    @GetMapping
    public String home() {
        return "UltraLedger2 is running!";
    }
    
    @GetMapping("/health")
    public String health() {
        return "OK";
    }
}

