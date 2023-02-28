import string
import sys
import math


def get_parameter_vectors():
    """
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    described in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    """
    # Implementing vectors e,s as lists (arrays) of length 26
    # with p[0] being the probability of 'A' and so on
    e = [0] * 26
    s = [0] * 26

    with open('e.txt', encoding='utf-8') as f:
        for line in f:
            # strip: removes the newline character
            # split: split the string on space character
            char, prob = line.strip().split(" ")
            # ord('E') gives the ASCII (integer) value of character 'E'
            # we then subtract it from 'A' to give array index
            # This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char) - ord('A')] = float(prob)
    f.close()

    with open('s.txt', encoding='utf-8') as f:
        for line in f:
            char, prob = line.strip().split(" ")
            s[ord(char) - ord('A')] = float(prob)
    f.close()

    return e, s


def shred(filename):
    # Using a dictionary here. You may change this to any data structure of
    # your choice such as lists (X=[]) etc. for the assignment
    X = dict.fromkeys(string.ascii_uppercase, 0)

    with open(filename, encoding='utf-8') as f:
        # TODO: add your code here
        for line in f:
            for letter in range(len(line) - 1):
                if line.strip()[letter].upper() in X:
                    X[line.strip()[letter].upper()] += 1
    return X


# TODO: add your code here for the assignment
def calculate_fvalue(x, limit, language, prob):
    return math.log(prob) + summation(x, limit, language)


def summation(x, limit, language):
    return (sum(x[chr(start_index + 65)] * math.log(get_parameter_vectors()[language][start_index])
                for start_index in range(0, limit)))


def conditional_prob(english_fval, spanish_fval):
    if spanish_fval - english_fval >= 100:
        return 0
    if spanish_fval - english_fval <= -100:
        return 1
    else:
        return 1 / (1 + pow(math.e, spanish_fval - english_fval))


english = 0
spanish = 1

english_prob = 0.6
spanish_prob = 0.4
# You are free to implement it as you wish!
# Happy Coding!

print("Q1")
x = shred("letter.txt")
for key, value in x.items():
    print(key, value)

print("Q2")

english1 = summation(x, 1, english)
spanish1 = summation(x, 1, spanish)
print(f'{english1:.4f}')
print(f'{spanish1:.4f}')

print("Q3")
english_fvalue = calculate_fvalue(x, 26, english, english_prob)
spanish_fvalue = calculate_fvalue(x, 26, spanish, spanish_prob)
print(f'{english_fvalue:.4f}')
print(f'{spanish_fvalue:.4f}')

print("Q4")
conditional_probability = conditional_prob(english_fvalue, spanish_fvalue)
print(f'{conditional_probability:.4f}')
