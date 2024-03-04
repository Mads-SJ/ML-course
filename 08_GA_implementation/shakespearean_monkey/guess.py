import random

target = "to be or not to be that is the question"

def generate_random_string():
    letters = "abcdefghijklmnopqrstuvxyz "
    string = ""
    for _ in range(len(target)):
        n = random.randint(0, len(letters) - 1)
        string += letters[n]
    return string

def generate_strings(number_of_strings):
    strings = []
    for i in range(number_of_strings):
        strings.append(generate_random_string())
    return strings

def fitness(string):
    fitness = 0
    for i in range(len(string)):
        if (string[i] == target[i]):
            fitness += 1
    return fitness


strings = generate_strings(100000)
best_fitness = 0
best_string = ""
for i in range(len(strings)):
    curr_fitness = fitness(strings[i])
    if curr_fitness > best_fitness:
        best_fitness = curr_fitness
        best_string = strings[i]
    
print(best_string, best_fitness)
print(target)