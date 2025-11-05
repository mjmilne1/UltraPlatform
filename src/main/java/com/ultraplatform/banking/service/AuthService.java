package com.ultraplatform.banking.service;

import com.ultraplatform.banking.dto.*;
import com.ultraplatform.banking.entity.User;
import com.ultraplatform.banking.repository.UserRepository;
import com.ultraplatform.banking.security.CustomUserDetailsService;
import com.ultraplatform.banking.security.JwtService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.time.Instant;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
@Transactional
public class AuthService {
    
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;
    private final AuthenticationManager authenticationManager;
    private final CustomUserDetailsService userDetailsService;
    
    public AuthResponse register(RegisterRequest request) {
        // Check if user already exists
        if (userRepository.existsByUsername(request.getUsername())) {
            throw new RuntimeException("Username already exists");
        }
        if (userRepository.existsByEmail(request.getEmail())) {
            throw new RuntimeException("Email already exists");
        }
        
        // Create new user
        User user = User.builder()
            .username(request.getUsername())
            .email(request.getEmail())
            .password(passwordEncoder.encode(request.getPassword()))
            .firstName(request.getFirstName())
            .lastName(request.getLastName())
            .status(User.UserStatus.ACTIVE)
            .roles(Set.of(User.Role.USER))
            .apiKey(generateApiKey())
            .build();
        
        user = userRepository.save(user);
        log.info("New user registered: {}", user.getUsername());
        
        // Generate tokens
        var userDetails = userDetailsService.loadUserByUsername(user.getUsername());
        String accessToken = jwtService.generateToken(userDetails);
        String refreshToken = jwtService.generateRefreshToken(userDetails);
        
        return AuthResponse.builder()
            .accessToken(accessToken)
            .refreshToken(refreshToken)
            .tokenType("Bearer")
            .expiresIn(86400L) // 24 hours
            .username(user.getUsername())
            .email(user.getEmail())
            .roles(user.getRoles().stream().map(Enum::name).collect(Collectors.toSet()))
            .build();
    }
    
    public AuthResponse login(LoginRequest request) {
        // Authenticate
        authenticationManager.authenticate(
            new UsernamePasswordAuthenticationToken(
                request.getUsername(),
                request.getPassword()
            )
        );
        
        // Get user
        User user = userRepository.findByUsername(request.getUsername())
            .orElseThrow(() -> new RuntimeException("User not found"));
        
        // Update last login
        user.setLastLogin(Instant.now());
        user.setFailedLoginAttempts(0);
        userRepository.save(user);
        
        // Generate tokens
        var userDetails = userDetailsService.loadUserByUsername(user.getUsername());
        String accessToken = jwtService.generateToken(userDetails);
        String refreshToken = jwtService.generateRefreshToken(userDetails);
        
        log.info("User logged in: {}", user.getUsername());
        
        return AuthResponse.builder()
            .accessToken(accessToken)
            .refreshToken(refreshToken)
            .tokenType("Bearer")
            .expiresIn(86400L)
            .username(user.getUsername())
            .email(user.getEmail())
            .roles(user.getRoles().stream().map(Enum::name).collect(Collectors.toSet()))
            .build();
    }
    
    public AuthResponse refresh(String refreshToken) {
        String token = refreshToken.replace("Bearer ", "");
        String username = jwtService.extractUsername(token);
        
        var userDetails = userDetailsService.loadUserByUsername(username);
        
        if (jwtService.validateToken(token, userDetails)) {
            String newAccessToken = jwtService.generateToken(userDetails);
            
            User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
            
            return AuthResponse.builder()
                .accessToken(newAccessToken)
                .refreshToken(token)
                .tokenType("Bearer")
                .expiresIn(86400L)
                .username(user.getUsername())
                .email(user.getEmail())
                .roles(user.getRoles().stream().map(Enum::name).collect(Collectors.toSet()))
                .build();
        }
        
        throw new RuntimeException("Invalid refresh token");
    }
    
    public void logout(String token) {
        // In a production system, you might want to blacklist the token
        log.info("User logged out");
    }
    
    public boolean verifyToken(String token) {
        try {
            String jwt = token.replace("Bearer ", "");
            String username = jwtService.extractUsername(jwt);
            var userDetails = userDetailsService.loadUserByUsername(username);
            return jwtService.validateToken(jwt, userDetails);
        } catch (Exception e) {
            return false;
        }
    }
    
    private String generateApiKey() {
        return UUID.randomUUID().toString().replace("-", "");
    }
}

