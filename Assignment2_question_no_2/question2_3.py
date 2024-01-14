def caesar_decipher(ciphertext, shift):
    plaintext = ""
    for char in ciphertext:
        if char.isalpha():
            if char.islower():
                # This line performs the decryption for a lowercase letter. It calculates the new character using the formula (original_character - shift + 26) % 26 to handle wrapping around the alphabet. The result is converted back to a character using chr and added to the plaintext.
                plaintext += chr((ord(char) - shift - ord('a') + 26) % 26 + ord('a'))
            else:
                plaintext += chr((ord(char) - shift - ord('A') + 26) % 26 + ord('A'))
        else:
            plaintext += char

    return plaintext

def caesar_decipher_brute_force(ciphertext):
    for shift in range(26):
        decrypted_text = caesar_decipher(ciphertext, shift)
        print(f"Shift: {shift}, Decrypted text: {decrypted_text}")

# Example usage
encrypted_text = "inevnoyr"
caesar_decipher_brute_force(encrypted_text)
