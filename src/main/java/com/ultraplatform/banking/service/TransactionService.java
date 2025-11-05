package com.ultraplatform.banking.service;

import com.ultraplatform.banking.entity.*;
import com.ultraplatform.banking.repository.*;
import com.ultraplatform.banking.dto.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;
import java.math.BigDecimal;
import java.time.Instant;

@Service
@RequiredArgsConstructor
@Slf4j
public class TransactionService {
    private final AccountRepository accountRepository;
    private final TransactionRepository transactionRepository;
    private final KafkaEventPublisher eventPublisher;
    private final EventSourcingService eventSourcingService;
    
    @Transactional(isolation = Isolation.SERIALIZABLE)
    public TransferResponse transferFunds(TransferRequest request) {
        // Publish transfer initiated event
        eventPublisher.publishTransactionEvent(
            Transaction.builder()
                .id(request.getFromAccountId())
                .amount(request.getAmount())
                .type("TRANSFER")
                .status("INITIATED")
                .build(),
            "TRANSFER_INITIATED"
        );
        
        Account fromAccount = accountRepository.findByIdWithLock(request.getFromAccountId())
            .orElseThrow(() -> new RuntimeException("Source account not found"));
        Account toAccount = accountRepository.findByIdWithLock(request.getToAccountId())
            .orElseThrow(() -> new RuntimeException("Target account not found"));
        
        if (fromAccount.getAvailableBalance().compareTo(request.getAmount()) < 0) {
            // Publish transfer failed event
            eventPublisher.publishTransactionEvent(
                Transaction.builder()
                    .accountId(request.getFromAccountId())
                    .amount(request.getAmount())
                    .status("FAILED")
                    .build(),
                "TRANSFER_FAILED"
            );
            throw new RuntimeException("Insufficient funds");
        }
        
        String txRef = "TXN-" + System.nanoTime();
        
        // Create debit transaction
        Transaction debit = Transaction.builder()
            .transactionRef(txRef + "-DR")
            .accountId(fromAccount.getId())
            .type("TRANSFER")
            .entryType("DEBIT")
            .amount(request.getAmount())
            .description(request.getDescription())
            .status("POSTED")
            .build();
        
        fromAccount.setBalance(fromAccount.getBalance().subtract(request.getAmount()));
        fromAccount.setAvailableBalance(fromAccount.getAvailableBalance().subtract(request.getAmount()));
        debit.setBalanceAfter(fromAccount.getBalance());
        
        // Create credit transaction
        Transaction credit = Transaction.builder()
            .transactionRef(txRef + "-CR")
            .accountId(toAccount.getId())
            .type("TRANSFER")
            .entryType("CREDIT")
            .amount(request.getAmount())
            .description(request.getDescription())
            .status("POSTED")
            .build();
        
        toAccount.setBalance(toAccount.getBalance().add(request.getAmount()));
        toAccount.setAvailableBalance(toAccount.getAvailableBalance().add(request.getAmount()));
        credit.setBalanceAfter(toAccount.getBalance());
        
        accountRepository.save(fromAccount);
        accountRepository.save(toAccount);
        Transaction savedDebit = transactionRepository.save(debit);
        Transaction savedCredit = transactionRepository.save(credit);
        
        // Publish events to Kafka
        eventPublisher.publishAccountEvent(fromAccount, "ACCOUNT_DEBITED");
        eventPublisher.publishAccountEvent(toAccount, "ACCOUNT_CREDITED");
        eventPublisher.publishTransactionEvent(savedDebit, "TRANSFER_COMPLETED");
        eventPublisher.publishTransactionEvent(savedCredit, "TRANSFER_COMPLETED");
        
        // Store in event sourcing
        eventSourcingService.saveEvent(
            fromAccount.getId(),
            "TRANSFER_COMPLETED",
            txRef,
            Instant.now()
        );
        
        log.info("Transfer completed and events published: {}", txRef);
        
        return TransferResponse.builder()
            .transactionRef(txRef)
            .amount(request.getAmount())
            .fromAccountId(fromAccount.getId())
            .toAccountId(toAccount.getId())
            .status("SUCCESS")
            .timestamp(Instant.now())
            .build();
    }
}

