use aes_gcm::{
    aead::{KeyInit, generic_array::GenericArray, generic_array::typenum::U16, Error as AeadError},
    Aes256Gcm,
    AeadInPlace,
};
use rand::RngCore;
use std::env;
use std::fs::File;
use std::io::Read;

fn generate_random_bytes(size: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    let mut result = vec![0u8; size];
    rng.fill_bytes(&mut result);
    result
}

fn encrypt(key: &[u8], iv: &[u8], plain_text: &mut [u8], additional_data: &[u8]) -> Result<GenericArray<u8, U16>, AeadError> {
    let key = GenericArray::from_slice(key);
    let cipher = Aes256Gcm::new(key);

    let nonce = GenericArray::from_slice(iv);
    
    // Encrypt the plain_text and return the authentication tag
    let tag = cipher.encrypt_in_place_detached(nonce, additional_data, plain_text)
        .expect("encryption failure");

    Ok(tag)
}

fn decrypt(key: &[u8], iv: &[u8], cipher_text: &mut [u8], additional_data: &[u8], tag: &GenericArray<u8, U16>) -> Result<(), AeadError> {
    let key = GenericArray::from_slice(key);
    let cipher = Aes256Gcm::new(key);

    let nonce = GenericArray::from_slice(iv);

    // Decrypt the cipher_text and verify the authentication tag
    Ok(cipher.decrypt_in_place_detached(nonce, additional_data, cipher_text, tag)?)
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        println!("Usage: {} <model>", args[0]);
        return;
    }

    let filename = &args[1];
    let mut file = File::open(filename).expect("File not found!");
    let mut plain_text = Vec::new();
    file.read_to_end(&mut plain_text).expect("Error in reading file!");

    let key = generate_random_bytes(32);
    let iv = generate_random_bytes(12);
    let additional_data = generate_random_bytes(64);

    println!("Key: {:?}", key);
    println!("IV: {:?}", iv);
    println!("Additional Data: {:?}", additional_data);

    //println!("Plaintext: {:?}", plain_text);
    match encrypt(&key, &iv, &mut plain_text, &additional_data) {
        Ok(tag) => {
            //println!("Ciphertext: {:?}", plain_text);

            match decrypt(&key, &iv, &mut plain_text, &additional_data, &tag) {
                Ok(_) => {
                    //println!("Plaintext: {:?}", plain_text);
                    println!("Decryption is correct!");
                },
                Err(_) => {
                    println!("Error in decryption process!");
                }
            }
        },
        Err(_) => {
            eprintln!("Error in encryption process!");
        }
    }

}
