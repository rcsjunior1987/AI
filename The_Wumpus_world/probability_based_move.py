
#----- IFN680 Assignment 1 -----------------------------------------------#
#  The Wumpus World: a probability based agent
#
#  Implementation of two functions
#   1. PitWumpus_probability_distribution()
#   2. next_room_prob()
#
#    Student no: n10374647
#    Student name: Roberto Carlos da Silva Junior
#
#-------------------------------------------------------------------------#
from random import *
from AIMA.logic import *
from AIMA.utils import *
from AIMA.probability import *
from tkinter import messagebox

#--------------------------------------------------------------------------------------------------------------
#
#  The following two functions are to be developed by you. They are functions in class Robot. If you need,
#  you can add more functions in this file. In this case, you need to link these functions at the beginning
#  of class Robot in the main program file the_wumpus_world.py.
#
#--------------------------------------------------------------------------------------------------------------
#   Function 1. PitWumpus_probability_distribution(self, width, height)
#
# For this assignment, we treat a pit and the wumpus equally. Each room has two states: 'empty' or 'containing a pit or the wumpus'.
# A Boolean variable to represent each room: 'True' means the room contains a pit/wumpus, 'False' means the room is empty.
#
# For a cave with n columns and m rows, there are totally n*m rooms, i.e., we have n*m Boolean variables to represent the rooms.
# A configuration of pits/wumpus in the cave is an event of these variables.
#
# The function PitWumpus_probability_distribution() below is to construct the joint probability distribution of all possible
# pits/wumpus configurations in a given cave, two parameters
#
# width : the number of columns in the cave
# height: the number of rows in the cave
#
# In this function, you need to create an object of JointProbDist to store the joint probability distribution and  
# return the object. The object will be used by your function next_room_prob() to calculate the required probabilities.
#
# This function will be called in the constructor of class Robot in the main program the_wumpus_world.py to construct the
# joint probability distribution object. Your function next_room_prob() will need to use the joint probability distribution
# to calculate the required conditional probabilities.
#
def PitWumpus_probability_distribution(self, width, height): 
       
    ##Get the available_rooms (Here called as frontier) 
    frontier = []
    frontier = list(self.available_rooms)
                            
    ##Get the room that are known and have breeze
    known_BS = self.observation_breeze_stench(self.visited_rooms)
    
    ##Get the room that are known and have Pit/Wumpus
    known_PW = self.observation_pits(self.visited_rooms)     
                                         
    ##Get all the possible events for the  all Posible Events
    ##Giving true to the room which supposed to has PW
    P = JointProbDist(frontier, { each:[T, F] for each in frontier })
    all_posible_events = all_events_jpd(frontier, P, known_PW)            
         
    ##Inside each item of all possible events     
    for each_possible_event in all_posible_events:                          
        prob_PW = 1
        ##analyse the probability of a room has PW to 𝜬(𝑃𝑞, 𝑃𝑊𝑘𝑛𝑜𝑤𝑛) 
        for (room, value) in each_possible_event.items():                
            ##if so it increments 0.2
            if value: prob_PW *= .2
            ##Otherwise it increments 0.8    
            else: prob_PW *= .8
            
            ##Obtain the value of prob_known_BS (𝜬(𝐵𝑆𝑘𝑛𝑜𝑤𝑛|𝑃𝑞,𝑃𝑊𝑘𝑛𝑜𝑤𝑛))
            if (self.consistent(known_BS, each_possible_event) == 0): prob_known_BS = 1
            else: prob_known_BS = self.consistent(known_BS, each_possible_event)                        

            ## Multiply the prob_known_BS (𝜬(𝐵𝑆𝑘𝑛𝑜𝑤𝑛|𝑃𝑞,𝑃𝑊𝑘𝑛𝑜𝑤𝑛)) and prob_PW (𝜬(𝑃𝑞, 𝑃𝑊𝑘𝑛𝑜𝑤𝑛))
            P[each_possible_event] = prob_known_BS * prob_PW
                                    
        return P           
                              
#---------------------------------------------------------------------------------------------------
#   Function 2. next_room_prob(self, x, y)
#
#  The parameters, (x, y), are the robot's current position in the cave environment.
#  x: column
#  y: row
#
#  This function returns a room location (column,row) for the robot to go.
#  There are three cases:
#
#    1. Firstly, you can call the function next_room() of the logic-based agent to find a
#       safe room. If there is a safe room, return the location (column,row) of the safe room.
#    2. If there is no safe room, this function needs to choose a room whose probability of containing
#       a pit/wumpus is lower than the pre-specified probability threshold, then return the location of
#       that room.
#    3. If the probabilities of all the surrounding rooms are not lower than the pre-specified probability
#       threshold, return (0,0).
#
def next_room_prob(self, x, y):
    ##messagebox.showinfo("Not yet complete", "You need to complete the function next_room_prob.")
    ##pass
    
    # Declares the variable new_room which indicates which is the next room 
    # that the agente is moving to
    new_room = (0, 0) 
    lowest_prob = 1    

    # Get the size of the board
    n = self.cave.WIDTH,
    m = self.cave.HEIGHT                    
    
    # Declare the variable which is used to calculate the probability of a room has
    # P/W
    prob_each_room_has_PW = 0
    
    ##Get the room that are known and have breeze
    known_BS = self.observation_breeze_stench(self.visited_rooms)
    
    #Analyze each available room
    for each_room in frontier:
        #if this room is known and safe
        if self.check_safety(each_room[0], each_room[1]) == True:
            #The agent is moved to it
            new_room = each_room                
            break
        #Otherwise
        else:                              
           
            #The probability of this room has P/W is got
            pRoom = enumerate_joint_ask(each_room, {}, self.PitWumpus_probability_distribution(n,m))
                      
            # If this probability is lower than the maximum probability described, the 
            # it it difined this room to be the next agent room
            if prob_each_room_has_PW < lowest_prob:
                lowest_prob = prob_each_room_has_PW
                if lowest_prob <= self.max_pit_probability: new_room = each_room
             
    # return the room which the agent is moving to                                                                                                    
    return new_room
    
#---------------------------------------------------------------------------------------------------
 
####################################################################################################
