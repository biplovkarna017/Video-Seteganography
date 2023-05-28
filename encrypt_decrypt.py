from base64 import b64encode, b64decode
import hashlib
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes

# define encrypt
def encrypt(plain_text, password):
    # generate a random salt
    salt = get_random_bytes(AES.block_size)

    # use the Scrypt KDF to get a private key from the password
    private_key = hashlib.scrypt(
        password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)

    # create cipher config
    cipher_config = AES.new(private_key, AES.MODE_GCM)

    # return a dictionary with the encrypted text
    cipher_text, tag = cipher_config.encrypt_and_digest(bytes(plain_text, 'utf-8'))
    return [
        b64encode(cipher_text).decode('utf-8'),
        b64encode(salt).decode('utf-8'),
        b64encode(cipher_config.nonce).decode('utf-8'),
        b64encode(tag).decode('utf-8')
    ]

# define decrypt
def decrypt(enc_list, password):
    # decode the dictionary entries from base64
    salt = b64decode(enc_list[1])
    cipher_text = b64decode(enc_list[0])
    nonce = b64decode(enc_list[2])
    tag = b64decode(enc_list[3])
    

    # generate the private key from the password and salt
    private_key = hashlib.scrypt(
        password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)

    # create the cipher config
    cipher = AES.new(private_key, AES.MODE_GCM, nonce=nonce)

    # decrypt the cipher text
    decrypted = cipher.decrypt_and_verify(cipher_text, tag)

    return decrypted

