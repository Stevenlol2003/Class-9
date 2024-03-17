import sys
import math
import numpy as np

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    X = dict()
    with open (filename, encoding = 'utf-8') as f:
        for line in f:
            for char in line:
                char = char.upper()
                if char in alphabet:
                    if char in X:
                        X[char] += 1
                    else: 
                        X[char] = 1
    return X



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

# Section for Q1
letters = shred("letter.txt")
print("Q1")
for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    count = letters.get(letter, 0)
    print(letter + " " + str(count))

# Converting dictionary of number of alphabet to vector
letterVector = [0] * 26
for letter, count in letters.items():
    # ord() function learned from https://www.programiz.com/python-programming/methods/built-in/ord
    index = ord(letter) - ord('A')  # Calculating the index in vector by finding difference from A
    letterVector[index] = count
# print function used for debugging    
# print(letterVector)    

# Declares probability for English and Spanish
probEnglish = 0.6
probSpanish = 0.4

# Read and and store values for e and s by calling given function
e, s = get_parameter_vectors()

# Section for Q2
print("Q2")
X1 = letterVector[0] 
e1 = e[0]
s1 = s[0]
# round() function learned from https://www.w3schools.com/python/ref_func_round.asp
resulte1 = round((X1 * math.log(e1)), 4)
results1 = round((X1 * math.log(s1)), 4)
print(resulte1)
print(results1)

# Section for Q3
logProbEnglish = math.log(probEnglish)
logProbSpanish = math.log(probSpanish)
sumEnglish = 0
sumSpanish = 0

# Similar to Q2 except adding up the (number of letter appearance) * log(e or s value respectively)
for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    index = ord(char) - ord('A')
    Xchar = letterVector[index]
    echar = e[index]
    schar = s[index]
    
    sumEnglish += Xchar * math.log(echar)
    sumSpanish += Xchar * math.log(schar)

FEnglish = round((logProbEnglish + sumEnglish), 4)
FSpanish = round((logProbSpanish + sumSpanish), 4)

print("Q3")
print(FEnglish)
print(FSpanish)

# Section for Q4
probability = 0

if FSpanish - FEnglish >= 100:
    probability = 0
elif FSpanish - FEnglish <= -100:
    probability = 1
else:
    # exp() function learned from https://www.tutorialspoint.com/python/number_exp.htm
    probability = round((1 / (1 + math.exp(FSpanish - FEnglish))), 4)

print("Q4")
print(probability)    