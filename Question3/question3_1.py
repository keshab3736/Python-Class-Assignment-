

#code to obtain the key for the cipher
total = 0
for i in range(5):
    for j in range(3):
        if (i + j) == 5:
            total +=(i+j)
        else:
            total -= (i-j)

counter = 0
while counter < 5:
    if total<13:
        total += 1
    elif total > 13:
        total -= 1
    else:
        counter+=2
        
print(total)
print(counter)



original_code = '''
tybony_inevnoyr = 100
zl_qvpg {'xrl1': 'inyhr1', 'xr12': 'inyhr2', 'xr13': 'inyhr3'}
qrs cebprff_ahzoref():
    tybony tybony_inevnoyr
    ybpny_inevnoyr = 5
    ahzoref= [1, 2, 3, 4, 5]
    juvyr ybpny_inevnoyr > 0:
        vs ybpny_inevnoyr % 2 == 0:
        ahzoref.erzbir(ybpny_inevnoyr)
        ybpny_inevnoyr -= 1
    erghea ahzoref
zl_frg = {1, 2, 3, 4, 5, 5, 4, 3, 2, 1}
erfhyg = cebprff_ahzoref(ahzoref=zl_frg)

qrs zbqvsl_qvpg():
    ybpny_inevnoyr = 10
    zl_qvpg['xr14'] = ybpny_inevnoyr

zbqvsl_qvpg(5)

qrs hcqngr_tybony():
    tybony tybony_inevnoyr
    tybony_inevnoyr += 10

sbe v va enatr(5):
    cevag(v)
    V += 1

vs zl_frg vf abg Abar naq zl_qvpg['xr14']==10:
    cevag("Pbaqvgvba zrgl")

vs 5 abg va zl_qvpg:
    cevag("5 abg sbhaq va gur qvpgvbanel!")

cevag(tybony_inevnoyr)
cevag(zl_qvpg)
cevag(zl_frg)
'''
## Code to decrypt the raw code
print("oritial code code to understandable code")
def encrypt(text, key):
    encrypted_text =""
    for char in text:
        if char.isalpha():
            shifted = ord(char) + key
            if char.islower():
          
                if shifted > ord('z'):
                    shifted -= 26
                elif shifted < ord('a'):
                    shifted + 26
            elif char.isupper():
                if shifted > ord('Z'):
         
                    shifted -= 26
                elif shifted< ord('A'):
                    shifted += 26
                    
            encrypted_text += chr(shifted)
        else:
            encrypted_text += char
    return encrypted_text

#from the code given, we found the key to be 13
key = 13
encrypted_code = encrypt (original_code, key)
print(encrypted_code)

## This is the code we obtained
'''
global_variable = 100

my_dict ={'key1': 'value1', 'ke12': 'value2', 'ke13': 'value3'}
def process_numbers():
    global global_variable
    local_variable = 5
    numbers= [1, 2, 3, 4, 5]
    while local_variable > 0:
        if local_variable % 2 == 0:
            numbers.remove(local_variable)
        local_variable -= 1
    return numbers
my_set = {1, 2, 3, 4, 5, 5, 4, 3, 2, 1}
result = process_numbers(numbers=my_set)

def modify_dict():
    local_variable = 10
    my_dict['ke14'] = local_variable

modify_dict(5)

def update_global():
    global global_variable
    global_variable += 10

for i in range(5):
    print(i)
    i += 1

if my_set is not None and my_dict['ke14']==10:
    print("Condition mety")

if 5 not in my_dict:
    print("5 not found in the dictionary!")

print(global_variable)
print(my_dict)
print(my_set)'''

##Fixing the above code
##Fixing the above code
##Fixing the above code

global_variable = 100
# Typo in dictionary key 'ke12' instead of 'key2'
my_dict = {'key1': 'value1', 'ke2': 'value2', 'ke3': 'value3'}

def process_numbers():
    global global_variable
    local_variable = 5
    numbers = [1, 2, 3, 4, 5]
    
    # Issue: Modifying a list while iterating over it can lead to unexpected results.
    for item in numbers[:]:  # Use a copy of the list
        if item % 2 == 0:
            numbers.remove(item)
    return numbers

my_set = {1, 2, 3, 4, 5, 5, 4, 3, 2, 1}
# Error: The function process_numbers() doesn't accept any arguments.
result = process_numbers()

def modify_dict():
    local_variable = 10
    # Issue: The key is 'ke4', but it was mistakenly written as 'ke14'
    my_dict['key4'] = local_variable

modify_dict()  # Removed the argument as the function doesn't take any

def update_global():
    global global_variable
    global_variable += 10

for i in range(5):
    print(i)
    # Error: You don't need to increment 'i' within the loop; it increments automatically.

if my_set is not None and my_dict['key4'] == 10:
    print("Condition met")

# Issue: 5 is in the dictionary as a key, not the value. Use '5 in my_dict.keys()' or 'my_dict.get(5)'.
if 5 not in my_dict:
    print("5 not found in the dictionary!")

print(global_variable)
print(my_dict)
print(my_set)
