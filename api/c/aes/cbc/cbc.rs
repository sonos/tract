use aes::Aes256;
use block_modes::block_padding::Pkcs7;
use block_modes::{BlockMode, Cbc};
use std::io::{Error, ErrorKind};
use generic_array::GenericArray;
use aes::cipher::{KeyInit, BlockEncrypt, BlockDecrypt};
use hex;

pub fn pad_pkcs7(message: &str, block_size: usize) -> String {
    let padding_size = block_size - (message.len() % block_size);
    let padding_char = padding_size as u8;
    format!("{}{}", message, padding_char.to_string().repeat(padding_size))
}

pub fn xor_bytes(bytes1: &[u8], bytes2: &[u8]) -> Vec<u8> {
    let min_len = std::cmp::min(bytes1.len(), bytes2.len());
    let mut result = Vec::with_capacity(min_len);
    for i in 0..min_len {
        result.push(bytes1[i] ^ bytes2[i]);
    }
    result
}


pub fn aes_256_cbc_encrypt(message: &str, key_str: &str, iv_str: &str) -> Result<String, Error> {
    if key_str.len() != 32 || iv_str.len() != 16 {
        return Err(Error::new(ErrorKind::InvalidInput, "Invalid key or IV length"));
    }

    let padded_message = pad_pkcs7(message, 16);
    let message_bytes = padded_message.as_bytes();
    let iv = iv_str.as_bytes().to_vec();
    let mut prev_block = iv.clone();
    let key = GenericArray::clone_from_slice(key_str.as_bytes());
    let cipher = Aes256::new(&key);

    let mut ciphertext: Vec<u8> = Vec::new();
    for i in (0..message_bytes.len()).step_by(16) {
        let xor_block = xor_bytes(&prev_block, &message_bytes[i..i + 16]);
        let mut block = GenericArray::clone_from_slice(&xor_block);
        
        cipher.encrypt_block(&mut block);
        ciphertext.extend_from_slice(block.as_slice());
        prev_block = ciphertext[ciphertext.len() - 16..].to_vec();
    }

    Ok(hex::encode(ciphertext))
}

pub fn aes_256_cbc_decrypt(cipher_hex: &str, key_str: &str, iv_str: &str) -> Result<String, Error> {
    if key_str.len() != 32 || iv_str.len() != 16 {
        return Err(Error::new(ErrorKind::InvalidInput, "Invalid key or IV length"));
    }

    let ciphertext = hex::decode(cipher_hex).unwrap();
    let iv = iv_str.as_bytes().to_vec();
    let mut prev_block = iv.clone();
    let key = GenericArray::clone_from_slice(key_str.as_bytes());
    let cipher = Aes256::new(&key);

    let mut plaintext: Vec<u8> = Vec::new();
    for i in (0..ciphertext.len()).step_by(16) {
        let mut block = GenericArray::clone_from_slice(&ciphertext[i..i + 16]);
        cipher.decrypt_block(&mut block);
        let xor_block = xor_bytes(&prev_block, block.as_slice());
        plaintext.extend_from_slice(&xor_block);
        prev_block = ciphertext[i..i + 16].to_vec();
    }

    // Remove PKCS7 padding
    let padding_size = *plaintext.last().unwrap() as usize;
    let plaintext_len = plaintext.len();
    if padding_size > 0 && plaintext_len >= padding_size {
        plaintext.truncate(plaintext_len - padding_size);
    }

    Ok(String::from_utf8_lossy(&plaintext).to_string())
}


fn main() {
    let key = "01234567890123456789012345678901"; // 32 bytes key
    let msg = "Hello, world!123";
    let iv = "0123456789012345"; // 16 bytes IV

    match aes_256_cbc_encrypt(msg, key, iv) {
        Ok(encrypted) => {
            println!("Encrypted: {}", encrypted);
    
            match aes_256_cbc_decrypt(&encrypted, key, iv) {
                Ok(decrypted) => println!("Decrypted: {}", decrypted),
                Err(e) => println!("Decryption error: {}", e),
            }
        }
        Err(e) => println!("Encryption error: {}", e),
    }
}