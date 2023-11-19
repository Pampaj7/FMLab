from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import scrypt


def generate_keypair():
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    return private_key, public_key


def generate_dh_keypair():
    private_key = RSA.generate(2048)
    public_key = private_key.publickey().export_key()
    return private_key, public_key


def encrypt_symmetric_key(symmetric_key, public_key):
    recipient_key = RSA.import_key(public_key)
    cipher_rsa = PKCS1_OAEP.new(recipient_key)
    encrypted_key = cipher_rsa.encrypt(symmetric_key)
    return encrypted_key


def simulate_dh_key_exchange(private_key, other_party_public_key):
    other_party_key = RSA.import_key(other_party_public_key)

    # Simulate Diffie-Hellman key exchange
    # Note: This is a simplified simulation and may not be suitable for real-world scenarios

    # Use PKCS1_OAEP for encryption
    cipher = PKCS1_OAEP.new(other_party_key)
    encrypted_secret = cipher.encrypt(b"")

    # Use PKCS1_OAEP for decryption
    cipher = PKCS1_OAEP.new(private_key)
    shared_secret_material = cipher.decrypt(encrypted_secret)

    return shared_secret_material


def decrypt_symmetric_key(encrypted_key, private_key):
    key = RSA.import_key(private_key)
    cipher_rsa = PKCS1_OAEP.new(key)
    symmetric_key = cipher_rsa.decrypt(encrypted_key)
    return symmetric_key


def encrypt_data(data, symmetric_key):
    cipher_aes = AES.new(symmetric_key, AES.MODE_EAX)
    ciphertext, tag = cipher_aes.encrypt_and_digest(data.encode())
    return ciphertext, cipher_aes.nonce, tag


def decrypt_data(ciphertext, nonce, tag, symmetric_key):
    cipher_aes = AES.new(symmetric_key, AES.MODE_EAX, nonce=nonce)
    decrypted_data = cipher_aes.decrypt_and_verify(ciphertext, tag)
    return decrypted_data.decode()


def derive_symmetric_key(secret_material):
    # Using scrypt as an example KDF, you might use a different one based on your requirements
    key_length = 16  # Adjust the key length based on your security requirements
    salt = get_random_bytes(16)  # Salt for added security
    # Adjust parameters based on your security requirements
    return scrypt(secret_material, salt, key_length, N=2**14, r=8, p=1)


# Example Usage
if __name__ == "__main__":
    # Alice and Bob generate their key pairs
    alice_private_key, alice_public_key = generate_dh_keypair()
    bob_private_key, bob_public_key = generate_dh_keypair()

    # Print the generated keys for educational purposes
    print("Alice's private key:", alice_private_key)
    print("Alice's public key:", alice_public_key)
    print("Bob's private key:", bob_private_key)
    print("Bob's public key:", bob_public_key)

    # Alice and Bob simulate Diffie-Hellman key exchange
    alice_shared_secret_material = simulate_dh_key_exchange(
        alice_private_key, bob_public_key)
    bob_shared_secret_material = simulate_dh_key_exchange(
        bob_private_key, alice_public_key)

    # Print the shared secret material for educational purposes
    print("Alice's shared secret material:", alice_shared_secret_material)
    print("Bob's shared secret material:", bob_shared_secret_material)

    # Alice and Bob derive symmetric keys using a KDF
    alice_symmetric_key = derive_symmetric_key(alice_shared_secret_material)
    bob_symmetric_key = derive_symmetric_key(bob_shared_secret_material)

    # Print the derived symmetric keys for educational purposes
    print("Alice's symmetric key:", alice_symmetric_key)
    print("Bob's symmetric key:", bob_symmetric_key)

    # Alice and Bob simulate Diffie-Hellman key exchange
    alice_shared_secret_material = simulate_dh_key_exchange(
        alice_private_key, bob_public_key)
    bob_shared_secret_material = simulate_dh_key_exchange(
        bob_private_key, alice_public_key)

    # Alice and Bob derive symmetric keys using a KDF
    alice_symmetric_key = derive_symmetric_key(alice_shared_secret_material)
    bob_symmetric_key = derive_symmetric_key(bob_shared_secret_material)

    # Alice encrypts the symmetric key with Bob's public key
    encrypted_key = encrypt_symmetric_key(alice_symmetric_key, bob_public_key)

    # Alice encrypts the actual data with the symmetric key
    plaintext_data = "Hello, Bob! This is a secure message."
    ciphertext, nonce, tag = encrypt_data(plaintext_data, alice_symmetric_key)

    # Alice sends encrypted_key, ciphertext, nonce, and tag to Bob

    # Bob decrypts the symmetric key using his private key
    decrypted_symmetric_key = decrypt_symmetric_key(
        encrypted_key, bob_private_key)

    # Bob decrypts the actual data using the symmetric key
    decrypted_data = decrypt_data(
        ciphertext, nonce, tag, decrypted_symmetric_key)

    print("Decrypted Message:", decrypted_data)
