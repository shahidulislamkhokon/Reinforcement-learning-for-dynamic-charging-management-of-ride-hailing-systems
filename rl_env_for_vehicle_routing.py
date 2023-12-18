import os

from vehicle4 import charger_df, sim_interval_minutes, vehicle_speed_kmph, calculate_remaining_travel_time, \
    find_nearest_charger, calculate_fare, calculate_distance, calculate_trip_revenue, haversine, calculate_travel_time, \
    assign_vehicle_to_customer, calculate_energy_needed, customer_df, vehicle_df

os.environ['GUROBI_HOME'] = 'C:\gurobi1002\win64'

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import random
from math import radians, cos, sin, asin, sqrt
from datetime import datetime, timedelta
import numpy as np

import math
import gym
from gym import spaces
from gurobipy import Model, GRB
import gurobipy as gp
from datetime import timedelta
import matplotlib.pyplot as plt


class VehicleRoutingEnv(gym.Env):
    def __init__(self, sim_interval_minutes, vehicle_speed_kmph, calculate_remaining_travel_time, find_nearest_charger,calculate_fare, calculate_distance,calculate_trip_revenue, haversine, calculate_travel_time, assign_vehicle_to_customer,
                 calculate_energy_needed, customer_df, vehicle_df, charger_df):
        self.sim_interval_minutes = sim_interval_minutes
        self.vehicle_speed_kmph = vehicle_speed_kmph
        self.calculate_remaining_travel_time = calculate_remaining_travel_time
        self.find_nearest_charger = find_nearest_charger
        self.calculate_energy_needed = calculate_energy_needed
        self.calculate_fare = calculate_fare
        self.calculate_distance = calculate_distance
        self.calculate_trip_revenue = calculate_trip_revenue
        self.haversine = haversine
        self.calculate_travel_time = calculate_travel_time
        self.assign_vehicle_to_customer = assign_vehicle_to_customer

        # Initialize other environment variables
        self.state = None  # You'd replace this with an actual representation of your initial state
        self.done = False  # A flag to indicate whether the episode is done

        # Initialize dataframes or any other data structures you need
        self.customer_df = customer_df  # replace with your initialization
        self.vehicle_df = vehicle_df  # replace with your initialization
        self.charger_df = charger_df  # replace with your initialization

        # Initialize any other variables needed in your environment

    # Calculate fare for each customer
    def calculate_fare(travel_time, distance, beta_0, beta_1, beta_2):
        return beta_0 + beta_1 * travel_time + beta_2 * distance

    # Function to calculate energy needed to complete a trip
    def calculate_energy_needed(trip_distance, vehicle_efficiency):
        # Vehicle efficiency is in kWh/km, representing how much energy the vehicle consumes per kilometer

        energy_needed = trip_distance * vehicle_efficiency
        return energy_needed

    # Function to calculate the Euclidean distance between two points
    def calculate_distance(lat1, lon1, lat2, lon2):
        return math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)

    def calculate_trip_revenue(trip_distance, fare_structure, charge_per_minute, charge_per_kilometer):
        # You can define your fare structure based on distance or other factors
        fare_per_km = 0.1  # Adjust according to your scenario
        trip_revenue = trip_distance * fare_per_km
        return trip_revenue

    def haversine(self, lat1, lon1, lat2, lon2):
        # Function to calculate the haversine distance between two points given their latitudes and longitudes
        R = 6371  # Radius of the earth in kilometers
        dLat = radians(lat2 - lat1)  # difference btw two points of latitude
        dLon = radians(lon2 - lon1)  # difference btw two points of longitude
        a = sin(dLat / 2) * sin(dLat / 2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon / 2) * sin(dLon / 2)
        c = 2 * asin(sqrt(a))  # angular separation between teh two points
        distance = R * c
        return distance

    # Function to find the index of the nearest charger
    def find_nearest_charger(self, vehicle_lat, vehicle_lon, charger_df):
        nearest_charger_idx = None
        min_distance = float('inf')

        for idx, row in charger_df.iterrows():
            charger_lat, charger_lon = row['latitude'], row['longitude']
            distance = self.haversine(vehicle_lat, vehicle_lon, charger_lat, charger_lon)

            if distance < min_distance:
                min_distance = distance
                nearest_charger_idx = idx

        return nearest_charger_idx

    def calculate_travel_time(pickup_datetime, dropoff_datetime):
        # Calculate travel time
        travel_time = (dropoff_datetime - pickup_datetime).total_seconds() / 60  # Convert to minutes
        return travel_time

    # Define a function to calculate the remaining travel time in minutes based on distance and speed
    def calculate_remaining_travel_time(self, distance_km, speed_kmph):
        return distance_km / speed_kmph * 60

    def assign_vehicle_to_customer(self, customer_row, vehicle_df, available_vehicles):
        unavailable_vehicles = []  # Initialize a list to track unavailable vehicles

        for vehicle_id in available_vehicles:
            vehicle_row = vehicle_df.loc[vehicle_df['id'] == vehicle_id].iloc[0]

            # Calculate distance between vehicle and customer pickup location
            distance_to_customer = self.haversine(vehicle_row['latitude'], vehicle_row['longitude'],
                                             customer_row['pickup_latitude'], customer_row['pickup_longitude'])

            # Calculate travel time to customer pickup location
            travel_time_to_customer = self.calculate_remaining_travel_time(distance_to_customer, self.vehicle_speed_kmph)

            # Check if the vehicle can reach the customer within the pickup window
            if travel_time_to_customer <= (customer_row['pickup_datetime'] - self.current_time).total_seconds() / 60:
                # Assign the vehicle to the customer
                vehicle_df.at[vehicle_row.name, 'availability'] = 'unavailable'
                vehicle_df.at[vehicle_row.name, 'remaining_travel_time'] = travel_time_to_customer
            else:
                # Vehicle is unavailable, add to the list of unavailable vehicles
                unavailable_vehicles.append(vehicle_id)

        # Remove unavailable vehicles from the available_vehicles list
        available_vehicles = [v for v in available_vehicles if v not in unavailable_vehicles]

        return available_vehicles, unavailable_vehicles

    # Calculate fare for each customer

    def initialize_customer_data(self):
        pickup_datetimes = [datetime(2023, 8, 1, 0, random.randint(0, 59), random.randint(0, 59)) for _ in range(20)]
        dropoff_datetimes = [pickup + timedelta(minutes=random.randint(1, 10)) for pickup in pickup_datetimes]

        charger_location = {}

        # Sample customer and vehicle DataFrames
        customer_data = {
            'id': list(range(1, 21)),
            'pickup_datetime': [pickup.strftime('%Y-%m-%d %H:%M:%S') for pickup in pickup_datetimes],
            'dropoff_datetime': [dropoff.strftime('%Y-%m-%d %H:%M:%S') for dropoff in dropoff_datetimes],
            'pickup_latitude': [round(random.uniform(40.0, 41.0), 6) for _ in range(20)],
            'pickup_longitude': [round(random.uniform(-74.0, -73.0), 6) for _ in range(20)],
            'dropoff_latitude': [round(random.uniform(40.0, 41.0), 6) for _ in range(20)],
            'dropoff_longitude': [round(random.uniform(-74.0, -73.0), 6) for _ in range(20)],
            'status': ['waiting'] * 20,
            'assigned_vehicle': [None] * 20
        }

        # add a column called distance with customer_data
        distances = []

        for i in range(20):
            lat1 = customer_data['pickup_latitude'][i]
            lon1 = customer_data['pickup_longitude'][i]
            lat2 = customer_data['dropoff_latitude'][i]
            lon2 = customer_data['dropoff_longitude'][i]
            distance = self.haversine(lat1, lon1, lat2, lon2)
            distances.append(distance)

        customer_data['distance'] = distances

        # average_speed = 40
        # customer_df['travel_time'] = customer_df['distance'] / average_speed  # This will give travel time in hours

        vehicle_data = {
            'id': [1, 2, 3, 4, 5],
            'pickup_datetime': ['2023-08-01 00:00:00', '2023-08-01 00:05:00', '2023-08-01 00:10:00', '2023-08-01 00:15:00',
                                '2023-08-01 00:20:00'],
            'energy': [80, 70, 90, 60, 75],
            'latitude': [40.5, 40.6, 40.4, 40.7, 40.3],
            'longitude': [-73.8, -73.9, -73.7, -73.95, -73.65],
            'availability': ['available'] * 5,
            'state_of_charge': [0.8, 0.7, 0.9, 0.6, 0.75]  # Added state_of_charge values (ranges from 0 to 1)
        }

        # Sample charger Data
        charger_data = {
            'charger_id': list(range(1, 11)),
            'latitude': [round(random.uniform(40.0, 41.0), 6) for _ in range(10)],
            'longitude': [round(random.uniform(-74.0, -73.0), 6) for _ in range(10)],
            'charger_capacity': [random.randint(2, 5) for _ in range(10)]
        }

        charger_df = pd.DataFrame(charger_data)
        customer_df = pd.DataFrame(customer_data)
        vehicle_df = pd.DataFrame(vehicle_data)

        ## add trave_time column in customer_df dataset
        average_speed = 40
        customer_df['travel_time'] = customer_df['distance'] / average_speed  # This will give travel time in hours

        # print("customer_df: ", len(customer_df))
        # print("vehicle_df_main: ", vehicle_df)

        # Convert 'pickup_datetime' column to datetime format for both customer and vehicle DataFrames
        customer_df['pickup_datetime'] = pd.to_datetime(customer_df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
        vehicle_df['pickup_datetime'] = pd.to_datetime(vehicle_df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
        customer_df['dropoff_datetime'] = pd.to_datetime(customer_df['dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')
        # vehicle_df['pickup_datetime'] = pd.to_datetime(vehicle_df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')

        # Parameters
        num_simulations = 1
        sim_interval_minutes = 1  # Interval in minutes
        simulation_duration_minutes = 100  # Total simulation duration in minutes
        vehicle_speed_kmph = 50  # Vehicle speed in kilometers per minute

        # Initialize vehicle energy levels from the 'energy' column in the vehicle DataFrame
        initial_vehicle_energy = {row['id']: row['energy'] for _, row in vehicle_df.iterrows()}
        e = initial_vehicle_energy.copy()
        e_min = 20

    def step(self, action):
        for sim in range(1, self.num_simulations + 1):
            print(f"Simulation {sim}:")

            # Initialize the current time
            current_time = self.customer_df['pickup_datetime'].min()
            # available_vehicles = list(vehicle_df[vehicle_df['availability'] == 'available']['id'])
            unavailable_vehicles = []

            K = []

            # print("K_availability: ", K)
            e_min = 30

            waiting_customers = []
            I = []

            for _ in range(self.simulation_duration_minutes):
                print(f"Current Time: {current_time}")
                print("\n\n")
                # Get the pickup requests within the current time interval
                current_interval_requests = self.customer_df[(self.customer_df['pickup_datetime'] >= current_time) & (
                            self.customer_df['pickup_datetime'] < current_time + timedelta(minutes=self.sim_interval_minutes))]

                customer_id = set(current_interval_requests['id'])
                # print("I", I)
                # K = set(e.keys())
                # print("K: ", K)
                # print("new df ", vehicle_df)
                # print("before: ", vehicle_df)
                for _, vehicle_row in vehicle_df.iterrows():
                    if vehicle_row['availability'] == 'available':
                        K.append(vehicle_row['id'])
                print("K: ", len(K))
                if len(customer_id) > len(K):
                    # print("Number of customers exceeds number of vehicles, assigning equal number of customers to vehicles")
                    I = list(customer_id)[:len(K)]  # Assign equal number of customers to vehicles
                    waiting_customers.extend(list(customer_id)[len(K):])
                    print("waiting_customers: ", waiting_customers)
                elif len(customer_id) < len(K) or len(customer_id) == 0:
                    # print("Fewer customer requests or no new requests, taking customers from waiting list")
                    I = list(customer_id)
                    num_customers_needed = len(K) - len(customer_id)
                    # print(f"{num_customers_needed} number of customers are served from waiting list")
                    I.extend(waiting_customers[:num_customers_needed])
                    # I.append(customers_from_waiting)
                    waiting_customers = waiting_customers[num_customers_needed:]
                    print("waiting_customers: ", waiting_customers)
                elif len(K) == 0:
                    print("Please wait for available vehicle")
                else:
                    I = list(customer_id)

                # print("K: ", K)
                tau = {(i, j): self.calculate_travel_time(self.customer_df.loc[self.customer_df['id'] == i, 'pickup_datetime'].iloc[0],
                                                     vehicle_df.loc[vehicle_df['id'] == j, 'pickup_datetime'].iloc[0])
                       for i in I for j in K}
                # print("tau: ",tau)
                phi_d = {i: random.randint(10, 40) for i in I}
                M = 1000  # Adjust this value as needed
                distance_data = {
                    row['id']: self.haversine(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'],
                                         row['dropoff_longitude'])
                    for _, row in self.customer_df.iterrows()}
                # print("distance_data: ", distance_data)
                theta_w = {i: 0 for i in I}

                # Serve customers with available vehicles
                available_vehicles = [j for j in self.e.keys() if self.e[j] >= e_min]

                # Print information for the current minute
                print(f"Total Requests: {len(current_interval_requests)}")

                assignment_outcome_df, customer_status_df, vehicle_df, available_vehicles = self.solve_vehicle_routing_assignment(
                    self.I, self.K, self.tau, self.theta_w, self.e, self.phi_d, self.e_min, self.M, self.distance_data, self.current_time, self.available_vehicles,
                    self.current_interval_requests, self.customer_df, vehicle_df, self.charger_df, action)

                current_time += timedelta(minutes=self.sim_interval_minutes)

                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', None)

                print("customer_status_df\n", customer_status_df)
                print("vehicle_df_next\n", vehicle_df)
                K = []

    def calculate_fare(self, travel_time, distance, beta_0, beta_1, beta_2):
        return beta_0 + beta_1 * travel_time + beta_2 * distance

    # Function to calculate energy needed to complete a trip
    def calculate_energy_needed(self, trip_distance, vehicle_efficiency):
        # Vehicle efficiency is in kWh/km, representing how much energy the vehicle consumes per kilometer

        energy_needed = trip_distance * vehicle_efficiency
        return energy_needed

    # Function to calculate the Euclidean distance between two points
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        return math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)

    def calculate_trip_revenue(self, trip_distance, fare_structure, charge_per_minute, charge_per_kilometer):
        # You can define your fare structure based on distance or other factors
        fare_per_km = 0.1  # Adjust according to your scenario
        trip_revenue = trip_distance * fare_per_km
        return trip_revenue

    def haversine(self, lat1, lon1, lat2, lon2):
        # Function to calculate the haversine distance between two points given their latitudes and longitudes
        R = 6371  # Radius of the earth in kilometers
        dLat = radians(lat2 - lat1)  # difference btw two points of latitude
        dLon = radians(lon2 - lon1)  # difference btw two points of longitude
        a = sin(dLat / 2) * sin(dLat / 2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon / 2) * sin(dLon / 2)
        c = 2 * asin(sqrt(a))  # angular separation between teh two points
        distance = R * c
        return distance

    # Function to find the index of the nearest charger
    def find_nearest_charger(self, vehicle_lat, vehicle_lon, charger_df):
        nearest_charger_idx = None
        min_distance = float('inf')

        for idx, row in charger_df.iterrows():
            charger_lat, charger_lon = row['latitude'], row['longitude']
            distance = self.haversine(vehicle_lat, vehicle_lon, charger_lat, charger_lon)

            if distance < min_distance:
                min_distance = distance
                nearest_charger_idx = idx

        return nearest_charger_idx

    def calculate_travel_time(self, pickup_datetime, dropoff_datetime):
        # Calculate travel time
        travel_time = (dropoff_datetime - pickup_datetime).total_seconds() / 60  # Convert to minutes
        return travel_time

    # Define a function to calculate the remaining travel time in minutes based on distance and speed
    def calculate_remaining_travel_time(self, distance_km, speed_kmph):
        return distance_km / speed_kmph * 60

    def assign_vehicle_to_customer(self, customer_row, vehicle_df, available_vehicles):
        unavailable_vehicles = []  # Initialize a list to track unavailable vehicles

        for vehicle_id in available_vehicles:
            vehicle_row = vehicle_df.loc[vehicle_df['id'] == vehicle_id].iloc[0]

            # Calculate distance between vehicle and customer pickup location
            distance_to_customer = self.haversine(vehicle_row['latitude'], vehicle_row['longitude'],
                                             customer_row['pickup_latitude'], customer_row['pickup_longitude'])

            # Calculate travel time to customer pickup location
            travel_time_to_customer = self.calculate_remaining_travel_time(distance_to_customer, self.vehicle_speed_kmph)

            # Check if the vehicle can reach the customer within the pickup window
            if travel_time_to_customer <= (customer_row['pickup_datetime'] - self.current_time).total_seconds() / 60:
                # Assign the vehicle to the customer
                vehicle_df.at[vehicle_row.name, 'availability'] = 'unavailable'
                vehicle_df.at[vehicle_row.name, 'remaining_travel_time'] = travel_time_to_customer
            else:
                # Vehicle is unavailable, add to the list of unavailable vehicles
                unavailable_vehicles.append(vehicle_id)

        # Remove unavailable vehicles from the available_vehicles list
        available_vehicles = [v for v in available_vehicles if v not in unavailable_vehicles]

        return available_vehicles, unavailable_vehicles

    def reset(self):
        # Reset environment state to the initial state
        self.state = None  # Define a proper initial state here
        self.current_step = 0
        return self.state

    # def step_activities(self, action):
    #     # Get the new state, reward, done using the `solve_vehicle_routing_assignment` method
    #     assignment_outcome, new_customer_df, new_vehicle_df, new_available_vehicles = self.solve_vehicle_routing_assignment(
    #         self.I, self.K, self.tau, self.theta_w, self.e, self.phi_d, self.e_min, self.M, self.distance_data,
    #         self.current_time, self.available_vehicles,
    #         self.current_interval_requests, self.customer_df, self.vehicle_df, self.charger_df)
    #
    #     # Update the environment state
    #     self.state = {
    #         'assignment_outcome': assignment_outcome,
    #         'new_customer_df': new_customer_df.to_dict(orient='records'),
    #         'new_vehicle_df': new_vehicle_df.to_dict(orient='records'),
    #         'new_available_vehicles': new_available_vehicles,
    #         'current_time': self.current_time
    #     }
    #
    #     # Define the reward function and whether the episode is done
    #     # Note: you'll need to define a suitable reward function based on your problem specifics
    #     reward = self.calculate_reward(assignment_outcome, new_customer_df, new_vehicle_df)
    #     print("reward: ", reward)
    #
    #     # Determine if the episode is done (for instance, when all customers have been served)
    #     done = all(new_customer_df['status'] == 'finished')
    #     print("Done: ",done)
    #
    #     self.current_step += 1
    #
    #     return self.state, reward, done, {}

    # def render(self):
    #     # Implement rendering of the environment's state
    #     pass

    def render(self):
        # Implement rendering of the environment's state
        # Get data (replace with your actual data retrieval code)
        customer_df = self.customer_df  # Your dataframe with customer data (replace with actual dataframe)
        vehicle_df = self.vehicle_df  # Your dataframe with vehicle data (replace with actual dataframe)

        # Get positions of all customers and vehicles
        customer_positions = customer_df[['pickup_latitude', 'pickup_longitude']].values
        vehicle_positions = vehicle_df[['latitude', 'longitude']].values

        # Plot customers (in red) and vehicles (in green)
        plt.scatter(customer_positions[:, 1], customer_positions[:, 0], c='red', label='Customers')
        plt.scatter(vehicle_positions[:, 1], vehicle_positions[:, 0], c='green', label='Vehicles')

        # Additional plot settings
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Vehicle Routing Environment')
        plt.legend()

        # Show the plot
        plt.show()

    def solve_vehicle_routing_assignment(self, I, K, tau, theta_w, e, phi_d, e_min, M, distance_data, current_time,
                                         available_vehicles, current_interval_requests, customer_df, vehicle_df,
                                         charger_df):
        # Create a new Gurobi model
        model = gp.Model("Vehicle_Routing_Assignment")
        served_customers = []
        customer_df['status'] = customer_df['status'].str.strip()

        # Decision variables
        x = {}  # Binary decision variables: x[i, j] = 1 if customer i is assigned to vehicle j, 0 otherwise
        for i in I:
            for j in K:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # Update the model to incorporate new variables
        model.update()

        # Objective function: minimize the weighted sum of vehicle travel times and customer waiting times
        model.setObjective(gp.quicksum((tau[i, j] + theta_w[i]) * x[i, j] for i in I for j in K), GRB.MINIMIZE)

        # Constraints: Each customer must be assigned to exactly one vehicle
        for i in I:
            model.addConstr(gp.quicksum(x[i, j] for j in K) == 1)

        for j in K:
            # Each vehicle can serve at most one customer
            model.addConstr(gp.quicksum(x[i, j] for i in I) <= 1)

        # Energy constraint
        for i in I:
            for j in K:
                model.addConstr(e[j] >= phi_d[i] + e_min + M * (x[i, j] - 1))

        # Optimize the model
        model.optimize()

        # Check if the optimization was successful
        if model.status != GRB.OPTIMAL:
            print("Optimization failed!")
            return None

        # Create a dictionary to store the assignment outcome
        assignment_outcome = {}

        for i in I:
            assigned_vehicle = None
            for j in K:
                if x[i, j].X == 1:
                    assigned_vehicle = j
                    # print("assigned_vehicle: ", assigned_vehicle)
                    print(f"Customer {i} is assigned to Vehicle {j}")
                    break

            if assigned_vehicle is not None:
                assignment_outcome[i] = assigned_vehicle
                print("assignment_outcome", assignment_outcome)

        current_time += timedelta(minutes=self.sim_interval_minutes)

        vehicle_idx = None
        for _, customer_row in current_interval_requests.iterrows():
            if customer_row['status'] == 'waiting':
                if customer_row['id'] in assignment_outcome.keys():
                    assigned_vehicle_id = assignment_outcome[customer_row['id']]
                    vehicle_idx = assigned_vehicle_id - 1  # Adjust for 0-based indexing
                    if assigned_vehicle_id in available_vehicles:
                        customer_df.at[customer_row.name, 'status'] = 'assigned'
                        customer_df.at[customer_row.name, 'assigned_vehicle'] = assigned_vehicle_id
                        available_vehicles.remove(assigned_vehicle_id)
                        self.unavailable_vehicles.append(assigned_vehicle_id)
                        vehicle_row = vehicle_df[vehicle_df['id'] == assigned_vehicle_id].iloc[0]
                        customer_dropoff_distance = distance_data[customer_row['id']]
                        remaining_travel_time = self.calculate_remaining_travel_time(customer_dropoff_distance,
                                                                                self.vehicle_speed_kmph)
                        vehicle_df.at[vehicle_idx, 'availability'] = 'unavailable'
                        vehicle_df.at[vehicle_idx, 'remaining_travel_time'] = remaining_travel_time

                        # Remove the assigned vehicle from available_vehicles list
                        if assigned_vehicle_id in available_vehicles:
                            available_vehicles.remove(assigned_vehicle_id)

            # Check if SOC <= 25% for assigned vehicle
            if vehicle_df.at[vehicle_idx, 'state_of_charge'] <= 25:
                # Find nearest charger and calculate energy needed
                nearest_charger_idx = self.find_nearest_charger(vehicle_df.at[vehicle_idx, 'latitude'],
                                                                vehicle_df.at[vehicle_idx, 'longitude'], charger_df)
                energy_needed = self.calculate_energy_needed(vehicle_df.at[vehicle_idx, 'state_of_charge'], 90)

                # Calculate charging time and update SOC
                charger_power_kWh = 2
                charging_time_hours = energy_needed / charger_power_kWh
                vehicle_df.at[vehicle_idx, 'state_of_charge'] += charging_time_hours * charger_power_kWh

        for _, customer_row in customer_df.iterrows():
            if customer_row['dropoff_datetime'] <= current_time:
                customer_df.at[customer_row.name, 'status'] = 'finished'
                # customer_df.at[customer_row.name, 'assigned_vehicle'] = done
            elif customer_row['pickup_datetime'] <= current_time and customer_row['status'] == 'waiting':
                customer_df.at[customer_row.name, 'status'] = 'assigned'
                # customer_df.loc[customer_df['id'] == customer_row['id'], 'assigned_vehicle'] = j

        for vehicle_id, vehicle_row in vehicle_df.iterrows():
            assigned_customers = customer_df[customer_df['assigned_vehicle'] == vehicle_row['id']]

            if assigned_customers.empty:
                vehicle_df.at[vehicle_id, 'availability'] = 'available'
                vehicle_df.at[vehicle_id, 'remaining_travel_time'] = None
            else:
                # Check if all assigned customers have finished status
                if all(assigned_customers['status'] == 'finished'):
                    vehicle_df.at[vehicle_id, 'availability'] = 'available'
                    vehicle_df.at[vehicle_id, 'remaining_travel_time'] = None

                    # Make the vehicle available for the next assignment
                    available_vehicles.append(vehicle_row['id'])

        ## calculating revenue for each customer
        beta_0 = 2  # base fare
        beta_1 = 2  # charge per minutes
        beta_2 = 3  # charge per kilometer

        # Calculate revenue based on trip details and fare structure
        def compute_fare(row):
            if row['status'] == 'finished':
                # Assuming travel_time is in hours, so multiplying by 60 to get minutes.
                fare = beta_0 + (beta_1 * row['travel_time'] * 60) + (beta_2 * row['distance'])
                return fare
            else:
                return None  # If status is not "finished", then no fare.

        customer_df['fare'] = customer_df.apply(compute_fare, axis=1)

        # Print or store total_revenue
        # print("Total Revenue: ", total_revenue)

        return assignment_outcome, customer_df, vehicle_df, available_vehicles
env = VehicleRoutingEnv(sim_interval_minutes, vehicle_speed_kmph, calculate_remaining_travel_time, find_nearest_charger,calculate_fare, calculate_distance,calculate_trip_revenue, haversine, calculate_travel_time, assign_vehicle_to_customer,
                 calculate_energy_needed, customer_df, vehicle_df, charger_df)
print(env.solve_vehicle_routing_assignment)
env.render()
