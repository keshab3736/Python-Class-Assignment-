alpha_num = "v7sHnT2dQk1Xm4rA9lL3iO6zW8pUyB0"

number_string = ''
letter_string = ''
even_number_ascii = {}
upper_letter_ascii = {}

#iterating through the alpha_num string
for i in alpha_num:
    if i.isalpha():
        letter_string+=i
    elif i.isdigit():
        number_string+=i
        
for i in number_string:
    #changin the type of i and checking if it is an even number
    if int(i) % 2 ==0:
        # assign the key(number) values(ascii) to the dictionary
        even_number_ascii[i] = ord(i)
        
for i in letter_string:
    #checking if the letter is uppercase and assigning it to the dictionary
    if i.isupper():
        upper_letter_ascii[i]= ord(i)
        
print("The number string is: ",number_string)
print("The letter string is: ",letter_string)

print("The even number ascii values are: ",even_number_ascii)
print("The upper case letter ascii values are: ",upper_letter_ascii)