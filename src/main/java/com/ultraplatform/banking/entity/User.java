package com.ultraplatform.banking.entity;

import lombok.*;
import jakarta.persistence.*;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;
import java.time.Instant;

@Entity
@Table(name = "users")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class User {
    @Id
    @GeneratedValue
    private UUID id;
    
    @Column(unique = true, nullable = false)
    private String username;
    
    @Column(unique = true, nullable = false)
    private String email;
    
    @Column(nullable = false)
    private String password;
    
    private String firstName;
    private String lastName;
    
    @Enumerated(EnumType.STRING)
    @Builder.Default
    private UserStatus status = UserStatus.ACTIVE;
    
    @ElementCollection(fetch = FetchType.EAGER)
    @Enumerated(EnumType.STRING)
    @Builder.Default
    private Set<Role> roles = new HashSet<>();
    
    private String apiKey;
    private Instant lastLogin;
    private int failedLoginAttempts;
    
    @Column(updatable = false)
    @Builder.Default
    private Instant createdAt = Instant.now();
    
    public enum Role {
        ADMIN, MANAGER, USER, AUDITOR, API_CLIENT
    }
    
    public enum UserStatus {
        ACTIVE, SUSPENDED, LOCKED, PENDING_VERIFICATION
    }
}

