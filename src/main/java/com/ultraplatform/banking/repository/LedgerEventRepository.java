package com.ultraplatform.banking.repository;

import com.ultraplatform.banking.entity.LedgerEvent;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Repository
public interface LedgerEventRepository extends JpaRepository<LedgerEvent, UUID> {
    
    Optional<LedgerEvent> findTopByAggregateIdOrderBySequenceNumberDesc(UUID aggregateId);
    
    List<LedgerEvent> findByAggregateIdAndValidFromLessThanEqualOrderBySequenceNumber(
        UUID aggregateId, Instant validTime);
    
    @Query("SELECT e FROM LedgerEvent e WHERE e.aggregateId = ?1 " +
           "AND e.validFrom <= ?2 AND (e.validTo IS NULL OR e.validTo > ?2) " +
           "AND e.transactionTime <= ?3 ORDER BY e.sequenceNumber")
    List<LedgerEvent> findBitemporalEvents(UUID aggregateId, 
                                          Instant validTime, 
                                          Instant transactionTime);
}

