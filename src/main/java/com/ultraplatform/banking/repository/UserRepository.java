package com.ultraplatform.banking.repository;

import com.ultraplatform.banking.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;
import java.time.Instant;
import java.util.Optional;
import java.util.UUID;

@Repository
public interface UserRepository extends JpaRepository<User, UUID> {
    Optional<User> findByUsername(String username);
    Optional<User> findByEmail(String email);
    Optional<User> findByApiKey(String apiKey);
    Boolean existsByUsername(String username);
    Boolean existsByEmail(String email);
    
    @Modifying
    @Query("UPDATE User u SET u.lastLogin = ?2 WHERE u.id = ?1")
    void updateLastLogin(UUID userId, Instant lastLogin);
    
    @Modifying
    @Query("UPDATE User u SET u.failedLoginAttempts = ?2 WHERE u.id = ?1")
    void updateFailedAttempts(UUID userId, int attempts);
}

