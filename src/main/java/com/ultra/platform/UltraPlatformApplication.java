package com.ultra.platform;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.web.client.RestTemplate;

@SpringBootApplication
public class UltraPlatformApplication {
    public static void main(String[] args) {
        SpringApplication.run(UltraPlatformApplication.class, args);
        System.out.println("UltraPlatform started with UltraLedger2 integration!");
    }
    
    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
