import pandas as pd
import math
import itertools

data = pd.read_csv(r'C:\Users\Mads\Desktop\datamatiker\4. semester\ML\ML-code\08_GA_implementation\travelling_salesman\TSPcities1000.txt',sep='\s+',header=None)
data = pd.DataFrame(data)

import matplotlib.pyplot as plt
x = data[1]
y = data[2]

def distancebetweenCities(city1x, city1y, city2x, city2y):
    xDistance = abs(city1x-city2x)
    yDistance = abs(city1y-city2y)
    distance = math.sqrt((xDistance ** 2) + (yDistance ** 2))
    return distance

def calculateTotalDistance(route):
    totalDistance = 0
    for i in range(0, len(route) - 1):
        city1 = route[i]
        city2 = route[i+1]
        totalDistance += distancebetweenCities(x[city1], y[city1], x[city2], y[city2])
    return totalDistance

def brute_force(number_of_cities):
    cities = [i for i in range(number_of_cities)]
    all_possible_routes = [list(permutation) for permutation 
                           in list(itertools.permutations(cities))]
    print(all_possible_routes)
    shortest_route = []
    shortest_distance = 1e20
    for i in range(len(all_possible_routes)):
        route = all_possible_routes[i]
        distance = calculateTotalDistance(route)
        if distance < shortest_distance:
            shortest_distance = distance
            shortest_route = route
        print(i / len(all_possible_routes) * 100, "% done")

    return shortest_route, shortest_distance

print(brute_force(8))
# 8! = 40.320 ruter - tager 20-30 sekunder
# 9! = 362.880 ruter - tager 1-2 minutter
# 10! = 3.6 mio. ruter - tager 20-30 minutter