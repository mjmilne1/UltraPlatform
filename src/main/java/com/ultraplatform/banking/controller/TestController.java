package com.ultraplatform.banking.controller;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/test")
public class TestController {
    
    @GetMapping("/ping")
    public String ping() {
        return "System is running on PostgreSQL!";
    }
}

