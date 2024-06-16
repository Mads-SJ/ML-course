import random
import math
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

def find_elites(generation, percentage):
    sorted_gen = sorted(generation, key=fitness, reverse=True)
    return sorted_gen[:math.floor(len(sorted_gen) * percentage)]

def mutate(string):
    n = random.randint(0, len(string) - 1)
    new_letter = letters[random.randint(0, len(letters) - 1)]
    
    # Convert the string to a list to allow modification
    string_list = list(string)
    string_list[n] = new_letter
    
    # Convert the list back to a string
    new_string = ''.join(string_list)
    
    return new_string

def breed(mum, dad):
    # random split er overraskende meget bedre end at splitte 50/50
    split_index = random.randint(0, len(target) - 1)
    # split_index = 19
    mum_genes = ''.join(list(mum)[:split_index])
    dad_genes = ''.join(list(dad)[split_index:])
    child = mum_genes + dad_genes
    if random.random() < mutation_rate:
        child = mutate(child)
    return child



def crossover(generation):
    elites = find_elites(generation, 0.2)
    next_gen = []
    
    for i in range(len(generation)):
        mum = elites[random.randint(0, len(elites) - 1)]
        dad = elites[random.randint(0, len(elites) - 1)]
        next_gen.append(breed(mum, dad))
    return next_gen

def find_best_fitness(strings):
    best_fitness = 0
    best_string = ""
    for i in range(len(strings)):
        curr_fitness = fitness(strings[i])
        if curr_fitness > best_fitness:
            best_fitness = curr_fitness
            best_string = strings[i]
    return best_string, best_fitness


def evolve(size, epochs):
    current_gen = generate_strings(size)
    gen_counter = 0
    best_string, best_fitness = find_best_fitness(current_gen)
    for _ in range(epochs):
        current_gen = crossover(current_gen)
        best_string, best_fitness = find_best_fitness(current_gen)
        print(best_string, best_fitness)
        gen_counter += 1
        if (best_fitness == len(target)):
            print("Generations:", gen_counter)
            break
    return best_string

# measure time
start = time.time()
mutation_rate = 0.1
result = evolve(1000, 100)
print("Time:", time.time() - start, "seconds")