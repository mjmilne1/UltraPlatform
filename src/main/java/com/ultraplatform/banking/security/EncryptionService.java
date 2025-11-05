package com.ultraplatform.banking.security;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import javax.crypto.Cipher;
import javax.crypto.SecretKey;
import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.PBEKeySpec;
import javax.crypto.spec.SecretKeySpec;
import java.security.SecureRandom;
import java.security.spec.KeySpec;
import java.util.Base64;

@Service
public class EncryptionService {
    
    private static final String ALGORITHM = "AES/CBC/PKCS5Padding";
    private static final String KEY_ALGORITHM = "PBKDF2WithHmacSHA256";
    private static final int KEY_LENGTH = 256;
    private static final int ITERATION_COUNT = 65536;
    
    @Value("${encryption.master.key:DefaultMasterKey123}")
    private String masterKey;
    
    @Value("${encryption.salt:DefaultSalt456}")
    private String salt;
    
    public String encrypt(String plainText) throws Exception {
        SecureRandom random = new SecureRandom();
        byte[] iv = new byte[16];
        random.nextBytes(iv);
        IvParameterSpec ivSpec = new IvParameterSpec(iv);
        
        SecretKey key = generateKey();
        Cipher cipher = Cipher.getInstance(ALGORITHM);
        cipher.init(Cipher.ENCRYPT_MODE, key, ivSpec);
        
        byte[] cipherText = cipher.doFinal(plainText.getBytes("UTF-8"));
        byte[] combined = new byte[iv.length + cipherText.length];
        
        System.arraycopy(iv, 0, combined, 0, iv.length);
        System.arraycopy(cipherText, 0, combined, iv.length, cipherText.length);
        
        return Base64.getEncoder().encodeToString(combined);
    }
    
    public String decrypt(String encryptedText) throws Exception {
        byte[] combined = Base64.getDecoder().decode(encryptedText);
        byte[] iv = new byte[16];
        byte[] cipherText = new byte[combined.length - 16];
        
        System.arraycopy(combined, 0, iv, 0, iv.length);
        System.arraycopy(combined, iv.length, cipherText, 0, cipherText.length);
        
        IvParameterSpec ivSpec = new IvParameterSpec(iv);
        SecretKey key = generateKey();
        
        Cipher cipher = Cipher.getInstance(ALGORITHM);
        cipher.init(Cipher.DECRYPT_MODE, key, ivSpec);
        
        byte[] plainText = cipher.doFinal(cipherText);
        return new String(plainText, "UTF-8");
    }
    
    private SecretKey generateKey() throws Exception {
        SecretKeyFactory factory = SecretKeyFactory.getInstance(KEY_ALGORITHM);
        KeySpec spec = new PBEKeySpec(
            masterKey.toCharArray(),
            salt.getBytes(),
            ITERATION_COUNT,
            KEY_LENGTH
        );
        SecretKey tmp = factory.generateSecret(spec);
        return new SecretKeySpec(tmp.getEncoded(), "AES");
    }
    
    public String hashSensitiveData(String data) {
        try {
            java.security.MessageDigest md = java.security.MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest(data.getBytes("UTF-8"));
            return Base64.getEncoder().encodeToString(hash);
        } catch (Exception e) {
            throw new RuntimeException("Hashing failed", e);
        }
    }
}

