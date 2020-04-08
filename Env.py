# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 0 ..... m-1
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger
lambda_loc = [2, 12, 4, 7, 8]

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(i,j) for i in range(0,m) for j in range(0,m) if i!= j]
        self.action_space.append((0,0)) #Add action for no-ride scenario

        self.state_space = [(i, j, k) for i in range(0,m) for j in range(t) for k in range(d)]

        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. 
        This method converts a given state into a vector format. 
        Hint: The vector is of size m + t + d."""

        state_encod = [0 for _ in range(m+t+d)]
        state_encod[state[0]] = 1
        state_encod[m+state[1]] = 1
        state_encod[m+t+state[2]] = 1
        return state_encod


    # Use this function if you are using architecture-2 
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN.         This method converts a given state-action pair into a vector format. 
        Hint: The vector is of size m + t + d + m + m."""
        state_encod = [0 for _ in range(m+t+d+m+m)]
        state_encod[state[0]] = 1
        state_encod[m+state[1]] = 1
        state_encod[m+t+state[2]] = 1
        if (action[0] != 0 and action[1] != 0): #update only if not a 'no-ride' action
            state_encod[m+t+d+action[0]] = 1
            state_encod[m+t+d+m+action[1]] = 1

        return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location < len(lambda_loc): 
            requests = np.random.poisson(lambda_loc[location]) #get requests using lambda values given for each location
        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(0, (m-1)*m), requests) +  [(m-1)*m] # (0,0) is not considered as customer request so adding the last index where (0,0) is placed
        actions = [self.action_space[i] for i in possible_actions_index]

        return possible_actions_index,actions   



    def reward_func(self, time_to_arrive_at_pickup, ride_time, wait_time_till_next_slot):
        """Takes in state, action and Time-matrix and returns the reward"""
        reward = (R * ride_time) - (C * (time_to_arrive_at_pickup + ride_time + wait_time_till_next_slot))
        return reward


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state and time taken to travel to pickup and ride time and waiting time"""
        next_state = []
        
        curr_loc = state[0]
        curr_time_of_day = state[1]
        curr_day_of_week = state[2]
        pickup_loc = action[0]
        drop_loc = action[1]

        total_time   = 0
        time_to_arrive_at_pickup = 0
        wait_time_till_next_slot = 0
        ride_time = 0

        if ((pickup_loc== 0) and (drop_loc == 0)): # no-ride action, this means 1 hour wait time till next slot and next_loc is same as curr_loc
            wait_time_till_next_slot = 1
            next_loc = curr_loc
        elif (curr_loc == pickup_loc): # pickup location is same as driver's current location
            ride_time = Time_matrix[pickup_loc][drop_loc][curr_time_of_day][curr_day_of_week]
            next_loc = drop_loc
        else: # pickup location is different from driver's current location
            time_to_arrive_at_pickup = Time_matrix[curr_loc][pickup_loc][curr_time_of_day][curr_day_of_week]

            new_time_of_day, new_day_of_week = self.get_updated_time_day_to_reach_pickup(curr_time_of_day, curr_day_of_week, time_to_arrive_at_pickup)

            ride_time = Time_matrix[pickup_loc][drop_loc][new_time_of_day][new_day_of_week]
            next_loc  = drop_loc

        #Calculate new time after ride completion and calculate new time and day
        total_time = (wait_time_till_next_slot + time_to_arrive_at_pickup + ride_time)
        next_time_of_day, next_day_of_week = self.get_updated_time_day(curr_time_of_day, curr_day_of_week, total_time)

        return (next_loc, next_time_of_day, next_day_of_week), time_to_arrive_at_pickup, ride_time, wait_time_till_next_slot

    def get_updated_time_day_to_reach_pickup(self, time, day, ride_time ):
        """
        Takes in the current time and day and return new time and day after given ride_time.to reach pickup location
        """
        #if ride time is less than an hour, it is considered same hour while starting the ride, so return same time and day, else calculate new time
        if ride_time >= 1: 
            if (time + int(ride_time)) < 24: # Same day
                time = time + int(ride_time)
            else: # next day
                time = (time + int(ride_time)) % 24 
                num_of_days = (time + int(ride_time)) // 24
                day = (day + num_of_days ) % 7
        return time, day
    
    
    def get_updated_time_day(self, time, day, ride_time ):
        """
        Takes in the current time and day and return new time and day after given ride_time.
        """
        if (time + int(ride_time)) < 24: # Same day
            time = time + math.ceil(ride_time) #Since next ride is available at hourly interval, we take math.ceil to calculate the next request time
        else: # next day
            time = (time + math.ceil(ride_time)) % 24 
            num_of_days = (time + math.ceil(ride_time)) // 24
            day = (day + num_of_days ) % 7
        return time, day
    
    def step(self, state, action, Time_matrix):
        """
        Given current state and action, get next state, reward and total time for this step
        """
        next_state, time_to_arrive_at_pickup, ride_time, wait_time_till_next_slot = self.next_state_func(state, action, Time_matrix)
        reward = self.reward_func(time_to_arrive_at_pickup, ride_time, wait_time_till_next_slot)
        
        return next_state, reward, time_to_arrive_at_pickup+ride_time+wait_time_till_next_slot

    def reset(self):
        return self.action_space, self.state_space, self.state_init
