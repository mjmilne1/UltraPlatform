package com.ultraplatform.banking.dto;

import lombok.*;
import java.time.LocalDate;
import java.util.List;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AnalyticsReport {
    private String reportId;
    private String reportName;
    private ReportType type;
    private LocalDate startDate;
    private LocalDate endDate;
    private ReportMetrics metrics;
    private List<TransactionSummary> transactions;
    private Map<String, Object> additionalData;
    
    public enum ReportType {
        DAILY_SUMMARY,
        MONTHLY_STATEMENT,
        QUARTERLY_ANALYSIS,
        ANNUAL_REPORT,
        CUSTOM_RANGE
    }
}

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
class ReportMetrics {
    private Double totalInflow;
    private Double totalOutflow;
    private Double netBalance;
    private Integer transactionCount;
    private Double averageTransactionSize;
    private Double largestTransaction;
    private Map<String, Double> categoryBreakdown;
    private Map<String, Integer> dailyTransactionCounts;
    private Double growthRate;
    private List<TrendData> trends;
}

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
class TransactionSummary {
    private String transactionId;
    private String date;
    private String description;
    private Double amount;
    private String type;
    private String category;
    private String accountNumber;
}

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
class TrendData {
    private String metric;
    private List<DataPoint> dataPoints;
    private Double changePercentage;
    private String trend; // INCREASING, DECREASING, STABLE
}

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
class DataPoint {
    private String label;
    private Double value;
    private String timestamp;
}

