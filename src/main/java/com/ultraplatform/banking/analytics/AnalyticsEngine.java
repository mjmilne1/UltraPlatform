package com.ultraplatform.banking.analytics;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDate;
import java.util.*;

@Service
@Slf4j
public class AnalyticsEngine {
    
    @Autowired
    private JdbcTemplate jdbcTemplate;
    
    @Cacheable("analytics")
    public Map<String, Object> getDashboardMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        
        metrics.put("totalVolume", getTotalVolume());
        metrics.put("dailyAverage", getDailyAverageVolume());
        metrics.put("topAccounts", getTopAccountsByVolume());
        metrics.put("transactionTrends", getTransactionTrends());
        metrics.put("velocityMetrics", getVelocityMetrics());
        
        return metrics;
    }
    
    public BigDecimal getTotalVolume() {
        String sql = "SELECT COALESCE(SUM(amount), 0) FROM dw_transaction_fact WHERE transaction_date >= CURRENT_DATE - 30";
        return jdbcTemplate.queryForObject(sql, BigDecimal.class);
    }
    
    public Map<String, Object> getDailyAverageVolume() {
        String sql = """
            SELECT 
                AVG(daily_total) as avg_volume,
                STDDEV(daily_total) as stddev_volume,
                MAX(daily_total) as max_volume,
                MIN(daily_total) as min_volume
            FROM (
                SELECT DATE(transaction_date) as day, SUM(amount) as daily_total
                FROM dw_transaction_fact
                WHERE transaction_date >= CURRENT_DATE - 30
                GROUP BY DATE(transaction_date)
            ) daily_totals
        """;
        
        return jdbcTemplate.queryForMap(sql);
    }
    
    public List<Map<String, Object>> getTransactionTrends() {
        String sql = """
            SELECT 
                DATE(transaction_date) as date,
                COUNT(*) as transaction_count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount,
                COUNT(DISTINCT account_id) as unique_accounts
            FROM dw_transaction_fact
            WHERE transaction_date >= CURRENT_DATE - 30
            GROUP BY DATE(transaction_date)
            ORDER BY date
        """;
        
        return jdbcTemplate.queryForList(sql);
    }
    
    public Map<String, Object> getVelocityMetrics() {
        Map<String, Object> velocity = new HashMap<>();
        
        // Transactions per hour
        String hourlySQL = """
            SELECT hour_of_day, AVG(tx_count) as avg_per_hour
            FROM (
                SELECT hour_of_day, COUNT(*) as tx_count
                FROM dw_transaction_fact
                WHERE transaction_date >= CURRENT_DATE - 7
                GROUP BY DATE(transaction_date), hour_of_day
            ) hourly
            GROUP BY hour_of_day
            ORDER BY hour_of_day
        """;
        
        velocity.put("hourlyPattern", jdbcTemplate.queryForList(hourlySQL));
        
        // Peak hours
        String peakSQL = """
            SELECT hour_of_day, COUNT(*) as total
            FROM dw_transaction_fact
            WHERE transaction_date >= CURRENT_DATE - 7
            GROUP BY hour_of_day
            ORDER BY total DESC
            LIMIT 3
        """;
        
        velocity.put("peakHours", jdbcTemplate.queryForList(peakSQL));
        
        return velocity;
    }
    
    private List<Map<String, Object>> getTopAccountsByVolume() {
        String sql = """
            SELECT 
                a.account_name,
                COUNT(f.transaction_id) as transaction_count,
                SUM(f.amount) as total_volume
            FROM dw_transaction_fact f
            JOIN accounts a ON f.account_id = a.id
            WHERE f.transaction_date >= CURRENT_DATE - 30
            GROUP BY a.account_name
            ORDER BY total_volume DESC
            LIMIT 10
        """;
        
        return jdbcTemplate.queryForList(sql);
    }
}


