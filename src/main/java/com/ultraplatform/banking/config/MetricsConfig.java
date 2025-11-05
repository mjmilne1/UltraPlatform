package com.ultraplatform.banking.config;

import io.micrometer.core.aop.TimedAspect;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.config.MeterFilter;
import io.micrometer.prometheus.PrometheusConfig;
import io.micrometer.prometheus.PrometheusMeterRegistry;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.actuate.autoconfigure.metrics.MeterRegistryCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class MetricsConfig {
    
    @Value("${spring.application.name:ultraledger2}")
    private String applicationName;
    
    @Bean
    public MeterRegistryCustomizer<MeterRegistry> metricsCommonTags() {
        return registry -> registry.config()
            .commonTags(
                "application", applicationName,
                "environment", System.getProperty("spring.profiles.active", "default"),
                "region", System.getProperty("aws.region", "local"),
                "instance", System.getProperty("instance.id", "local-01")
            )
            .meterFilter(MeterFilter.deny(id -> {
                String uri = id.getTag("uri");
                return uri != null && (uri.startsWith("/actuator") || uri.startsWith("/swagger"));
            }));
    }
    
    @Bean
    public TimedAspect timedAspect(MeterRegistry registry) {
        return new TimedAspect(registry);
    }
    
    @Bean
    public PrometheusMeterRegistry prometheusMeterRegistry() {
        return new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
    }
}

