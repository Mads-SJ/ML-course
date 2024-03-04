import random
import time

target = "to be or not to be that is the question"
letters = "abcdefghijklmnopqrstuvxyz "

def generate_random_string():
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

def find_best_fitness(strings):
    best_fitness = 0
    best_string = ""
    for i in range(len(strings)):
        curr_fitness = fitness(strings[i])
        if curr_fitness > best_fitness:
            best_fitness = curr_fitness
            best_string = strings[i]
    return best_string, best_fitness

def find_neighbor(string):
    n = random.randint(0, len(string) - 1)
    new_letter = letters[random.randint(0, len(letters) - 1)]
    
    # Convert the string to a list to allow modification
    string_list = list(string)
    string_list[n] = new_letter
    
    # Convert the list back to a string
    new_string = ''.join(string_list)
    
    return new_string

# measure time 
start = time.time()
strings = generate_strings(100)
best_string, best_fitness = find_best_fitness(strings)
print(best_string, best_fitness)

n = 0
while(best_fitness < len(target)):
    neighbor = find_neighbor(best_string)
    neighbor_fitness = fitness(neighbor)
    if (neighbor_fitness > best_fitness):
        best_fitness = neighbor_fitness
        best_string = neighbor
        print(best_string, best_fitness)
    n += 1
print(n)
print("Time:", time.time() - start, "seconds")