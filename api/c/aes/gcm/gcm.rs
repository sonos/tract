use aes_gcm::{
    aead::{Aead, KeyInit, generic_array::GenericArray},
    Aes256Gcm,
};

fn encrypt(key: &[u8], iv: &[u8], plaintext: &[u8], additional_data: &[u8]) -> Vec<u8> {
    let key = GenericArray::from_slice(key);
    let cipher = Aes256Gcm::new(key);

    let nonce = GenericArray::from_slice(iv);
    
    let ciphertext = cipher.encrypt(nonce, plaintext, additional_data)
        .expect("encryption failure");

    let mut encrypted_data = ciphertext.to_vec();

    // Append the authentication tag to the encrypted data
    encrypted_data.extend_from_slice(cipher.tag().as_slice());

    encrypted_data
}

fn decrypt(key: &[u8], iv: &[u8], ciphertext: &[u8], additional_data: &[u8]) -> Option<Vec<u8>> {
    let key = GenericArray::from_slice(key);
    let cipher = Aes256Gcm::new(key);

    let nonce = GenericArray::from_slice(iv);
    
    // Extract the ciphertext and authentication tag
    let ciphertext_len = ciphertext.len();
    let tag_len = 12;
    let tag_start = ciphertext_len - tag_len;
    let encrypted_data = &ciphertext[..tag_start];
    let tag = &ciphertext[tag_start..];

    // Decrypt the ciphertext and verify the authentication tag
    cipher.decrypt(nonce, encrypted_data, tag, additional_data)
        .ok()
}

fn main() {
    let plaintext = b"backendengineer.io";
    let key = b"thiskeystrmustbe32charlongtowork";
    let iv = [0u8; 12]; // Initialize IV with 12 bytes (96 bits)
    let additional_data = b"Additional data for authentication";

    let ciphertext = encrypt(key, &iv, plaintext, additional_data);
    println!("Encrypted: {:?}", ciphertext);

    let decrypted_plaintext = decrypt(key, &iv, &ciphertext, additional_data)
        .expect("decryption failure");
    
    match String::from_utf8(decrypted_plaintext) {
        Ok(s) => {
            assert_eq!(s, "backendengineer.io");
            println!("Decryption successful!");
        },
        Err(_) => {
            println!("Decryption failed!");
        }
    }
}
