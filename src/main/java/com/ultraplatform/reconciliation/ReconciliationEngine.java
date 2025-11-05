package com.ultraplatform.banking.reconciliation;

import com.ultraplatform.banking.entity.Transaction;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Component
@Slf4j
public class ReconciliationEngine {
    
    @Autowired
    private ReconciliationAgent agent;
    
    private Map<String, BankRules> bankRules = new ConcurrentHashMap<>();
    
    public ReconciliationResult reconcile(LocalDate date, List<Transaction> ourTxns, List<ExternalTransaction> theirTxns) {
        ReconciliationResult result = new ReconciliationResult();
        result.date = date;
        result.totalOurs = ourTxns.size();
        result.totalTheirs = theirTxns.size();
        
        Map<Transaction, ExternalTransaction> matches = new HashMap<>();
        Set<ExternalTransaction> matched = new HashSet<>();
        
        for (Transaction ourTxn : ourTxns) {
            ExternalTransaction bestMatch = null;
            double bestScore = 0.0;
            
            for (ExternalTransaction theirTxn : theirTxns) {
                if (!matched.contains(theirTxn)) {
                    double score = calculateMatchScore(ourTxn, theirTxn);
                    if (score > bestScore) {
                        bestScore = score;
                        bestMatch = theirTxn;
                    }
                }
            }
            
            if (bestScore > 0.7) {
                matches.put(ourTxn, bestMatch);
                matched.add(bestMatch);
                result.matched++;
                
                // Check for discrepancies - FIX: Use toString() for UUID
                if (!ourTxn.getAmount().equals(bestMatch.getAmount())) {
                    result.addDiscrepancy(ourTxn.getId().toString(), "Amount mismatch");
                }
            } else {
                result.addUnmatched(ourTxn.getId().toString());
            }
        }
        
        // Find orphans in their transactions
        for (ExternalTransaction theirTxn : theirTxns) {
            if (!matched.contains(theirTxn)) {
                result.addOrphan(theirTxn.getId());
            }
        }
        
        return result;
    }
    
    private double calculateMatchScore(Transaction ours, ExternalTransaction theirs) {
        double score = 0.0;
        
        // Amount matching
        BigDecimal diff = ours.getAmount().subtract(theirs.getAmount()).abs();
        if (diff.compareTo(BigDecimal.ZERO) == 0) {
            score += 0.5;
        } else if (diff.compareTo(new BigDecimal("0.01")) <= 0) {
            score += 0.3;
        }
        
        // FIX: Use getDescription() instead of getReference()
        if (ours.getDescription() != null && ours.getDescription().equals(theirs.getReference())) {
            score += 0.3;
        }
        
        // FIX: Convert Instant to LocalDate properly
        LocalDate txnDate = ours.getCreatedAt().atZone(ZoneId.systemDefault()).toLocalDate();
        if (txnDate.equals(theirs.getDate())) {
            score += 0.2;
        }
        
        return score;
    }
    
    public static class ReconciliationResult {
        public LocalDate date;
        public int totalOurs;
        public int totalTheirs;
        public int matched;
        public List<String> unmatched = new ArrayList<>();
        public List<String> orphans = new ArrayList<>();
        public Map<String, String> discrepancies = new HashMap<>();
        
        public void addUnmatched(String id) { unmatched.add(id); }
        public void addOrphan(String id) { orphans.add(id); }
        public void addDiscrepancy(String id, String reason) { discrepancies.put(id, reason); }
    }
    
    public static class ExternalTransaction {
        public String id;
        public BigDecimal amount;
        public String reference;
        public LocalDate date;
        
        public String getId() { return id; }
        public BigDecimal getAmount() { return amount; }
        public String getReference() { return reference; }
        public LocalDate getDate() { return date; }
    }
    
    public static class BankRules {
        public BigDecimal amountTolerance = new BigDecimal("0.01");
        public long timeWindowMillis = 86400000; // 24 hours
    }
}

