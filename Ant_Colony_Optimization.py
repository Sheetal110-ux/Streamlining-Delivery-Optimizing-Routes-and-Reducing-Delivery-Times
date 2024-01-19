#implementing the Ant Colony Optimization (ACO) algorithm for the Traveling Salesman Problem (TSP) involves 
#modeling the behavior of ants to find an approximate solution. 
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt 

# Load latitude and longitude data from a CSV file
df = pd.read_csv('out_1.csv')

# Convert latitude and longitude columns to NumPy arrays
latitude = df['latitude'].values
longitude = df['longitude'].values

# Function to calculate Haversine distance between two coordinates
def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance

# Calculate the distance matrix using Haversine distance
n = len(latitude)
distance_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        distance_matrix[i][j] = haversine(latitude[i], longitude[i], latitude[j], longitude[j])

# ACO parameters
num_ants = 20
num_iterations = 100
alpha = 1.0  # Pheromone importance
beta = 2.0   # Distance importance
evaporation_rate = 0.5
pheromone_deposit = 100.0

# Initialize pheromone levels
pheromones = np.ones((n, n))

# Helper function to select the next city for an ant
def select_next_city(ant, pheromone, alpha, beta):
    current_city = ant[-1]
    unvisited_cities = [city for city in range(n) if city not in ant]

    probabilities = []
    total_prob = 0.0

    for city in unvisited_cities:
        distance_factor = 1.0 / ((distance_matrix[current_city][city] + 1e-6) ** beta)  # Add a small constant to avoid division by zero
        pheromone_factor = pheromone[current_city][city] ** alpha
        probability = pheromone_factor * distance_factor
        probabilities.append(probability)
        total_prob += probability

    # Normalize probabilities
    probabilities = [prob / total_prob for prob in probabilities]
    
    if total_prob == 0:
        # If total_prob is zero, assign equal probabilities to all unvisited cities
        equal_probability = 1.0 / len(unvisited_cities)
        probabilities = [equal_probability] * len(unvisited_cities)

    # Choose the next city based on probabilities
    next_city = random.choices(unvisited_cities, probabilities)[0]
    return next_city

# Initialize the best tour and its length
best_tour = None
best_tour_length = float('inf')

# Perform ACO iterations
for iteration in range(num_iterations):
    # Ants construct tours
    tours = []
    tour_lengths = []

    for ant in range(num_ants):
        tour = [random.randint(0, n - 1)]  # Start from a random city
        tour_length = 0.0

        while len(tour) < n:
            next_city = select_next_city(tour, pheromones, alpha, beta)
            tour.append(next_city)
            tour_length += distance_matrix[tour[-2]][next_city]

        # Return to the starting city to complete the tour
        tour_length += distance_matrix[tour[-1]][tour[0]]

        tours.append(tour)
        tour_lengths.append(tour_length)

    # Update the best tour if a shorter one is found
    if min(tour_lengths) < best_tour_length:
        best_tour_length = min(tour_lengths)
        best_tour = tours[tour_lengths.index(min(tour_lengths))]

    # Update pheromone levels
    pheromones *= (1 - evaporation_rate)  # Evaporation
    for ant, length in zip(tours, tour_lengths):
        for i in range(n - 1):
            pheromones[ant[i]][ant[i + 1]] += pheromone_deposit / length
        pheromones[ant[-1]][ant[0]] += pheromone_deposit / length

# Plot the best tour
best_tour_coordinates = [(latitude[i], longitude[i]) for i in best_tour]

# Add the starting point to the end to close the loop
best_tour_coordinates.append(best_tour_coordinates[0])

# Extract coordinates for plotting
tour_lats, tour_lons = zip(*best_tour_coordinates)

# Create a scatter plot of locations
plt.scatter(longitude, latitude, c='red', marker='o', label='Locations')

# Plot the best tour as a line
plt.plot(tour_lons, tour_lats, linestyle='-', linewidth=2, markersize=5, label='Best Tour', color='blue')

# Mark the starting point
plt.plot(tour_lons[0], tour_lats[0], 'go', label='Starting Point', markersize=8)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.title('Best Tour for TSP')
plt.grid(True)
plt.show()
