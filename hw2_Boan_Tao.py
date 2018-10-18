import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import math

# Question 1a ################################################################
'''
size of state space is width*length*12
assume width and length are 6
'''
up_head = [11, 0, 1]
down_head = [5, 6, 7]
right_head = [2, 3, 4]
left_head = [8, 9, 10]

Width = 6
Length = 6
State = np.zeros((Width, Length, 12))

# Question 1b ################################################################
'''
size of action space is 7
m = 0 no move        m = 1 forward          m = -1 backward
r = 0 no turn        r = 1 turn left        r = -1 turn right
'''

Action = np.zeros((2, 7))
Action = [(0, 0), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]


# Question 1c ################################################################
class Envir:

    def __init__(self, width, length, errorProbility):
        self.w = width
        self.l = length
        self.pe = errorProbility
        self.goal = (3,4)
        self.All_State_Situation = []
        for x in range(width):
            for y in range(length):
                for z in range(12):
                    self.All_State_Situation.append((x, y, z))


    # Probility of direction change
    def Direction_Probility(self, currentD, nextD, rotate):
        # calibrate rotation step
    
        Dchange = nextD - currentD
      
        # direction change = rotation(left/right/no)+error
        if Dchange == 0 + rotate:
            probD = 1 - 2 * self.pe
      
        elif Dchange == -1 + rotate or Dchange == 1 + rotate:
            probD = self.pe
        else:
            probD = 0
        return probD

    # Probility of moving to next block
    def Move_Probility(self, move, rotate, currentD, nextD, direction_set, opposite_direction_set):

        if move == 1 and (nextD - rotate)%12 in direction_set:
            probM = self.Direction_Probility(currentD, nextD, rotate)
        elif move == -1 and (nextD - rotate)%12 in opposite_direction_set:
            probM = self.Direction_Probility(currentD, nextD, rotate)
        else:
            probM = 0
            
        
        return probM

    # Calculate psa
    def psa(self, currentState, nextState, policy):

        current_x = currentState[0]
        current_y = currentState[1]
        currentD = currentState[2]
        next_x = nextState[0]
        next_y = nextState[1]
        nextD = nextState[2]
        move = policy[0]
        rotate = policy[1]
        

        # no move
        if current_x == next_x and current_y == next_y:

            # situation 1: no move command
            if move == 0:
                PSA = self.Direction_Probility(currentD, nextD, rotate)

            # situation 2: move off grid
            else:
                # top
                if current_x in range(1, self.w - 1, 1) and  current_y == self.l - 1:
                    PSA = self.Move_Probility(move, rotate, currentD, nextD, up_head , down_head)
                # bottom
                elif current_x in range(1, self.w - 1, 1) and current_y == 0:
                    PSA = self.Move_Probility(move, rotate, currentD, nextD, down_head , up_head)
                # right
                elif current_x == self.w - 1 and current_y in range(1, self.l - 1, 1):
                    PSA = self.Move_Probility(move, rotate, currentD, nextD, right_head, left_head)
                # left
                elif current_x == 0 and current_y in range(1, self.l - 1, 1):
                    PSA = self.Move_Probility(move, rotate, currentD, nextD, left_head, right_head)
                # bottom left corner (move down + left)
                elif current_x == 0 and current_y == 0:
                    PSA = self.Move_Probility(move, rotate, currentD, nextD, down_head , up_head) \
                           + self.Move_Probility(move, rotate, currentD, nextD, left_head, right_head)
                # top left corner (move up + left)
                elif current_x == 0 and current_y == self.l - 1:
                    PSA = self.Move_Probility(move, rotate, currentD, nextD, up_head, down_head) \
                           + self.Move_Probility(move, rotate, currentD, nextD, left_head, right_head)
                # bottom right corner (move down+right)
                elif current_x == self.w - 1 and current_y == 0:
                    PSA = self.Move_Probility(move, rotate, currentD, nextD, down_head, up_head) \
                           + self.Move_Probility(move, rotate, currentD, nextD, right_head, left_head)
                # top right corner (move up +right)
                elif current_x == self.w - 1 and current_y == self.l - 1:
                    PSA = self.Move_Probility(move, rotate, currentD, nextD, up_head, down_head) \
                           + self.Move_Probility(move, rotate, currentD, nextD, right_head, left_head)
                else:
                    PSA = 0
        # move up
        elif next_y == current_y+1 and next_x == current_x:
            PSA = self.Move_Probility(move, rotate, currentD, nextD, up_head, down_head)
        # move down
        elif next_y == current_y-1 and next_x == current_x:
            PSA = self.Move_Probility(move, rotate, currentD, nextD, down_head, up_head)
        # move left
        elif next_y == current_y and next_x == current_x-1:
            PSA = self.Move_Probility(move, rotate, currentD, nextD, left_head, right_head)
        # move right
        elif next_y == current_y and next_x == current_x+1:
            PSA = self.Move_Probility(move, rotate, currentD, nextD, right_head, left_head)

        return PSA

# Question 1d ################################################################

    # Position change
    def Position(self, head, move, x, y):

    
        # no move
        if move == 0:
            return x, y
        # move up
        elif (head in up_head and move == 1) or (head in down_head and move == -1):
            if y in range(0, self.w-1):
                y = y + 1
            else:
                pass
        # move down
        elif (head in down_head and move == 1) or (head in up_head and move == -1):
            if y in range(1, self.w):
                y = y - 1
            else:
                pass
        # move left
        elif (head in left_head and move == 1) or (head in right_head and move == -1):
            if x in range(1, self.l):
                x = x - 1
            else:
                pass
        # move right
        elif (head in right_head and move == 1) or (head in left_head and move == -1):
            if x in range(0,self.l-1):
                x = x + 1
            else:
                pass
        
        return x, y
    # give next state basing on PSA
    def nextState(self, currentState, policy, operation):

        current_x = currentState[0]
        current_y = currentState[1]
        current_head = currentState[2]
        
        move = policy[0]
        rotate = policy[1]
       
        head_posibility = [current_head, (current_head + 1)%12 , (current_head - 1)%12 ]

        
        # Possible situation  of next state
        next_state = []
       
        for DirectionProb in head_posibility:
           
            next_x, next_y = self.Position(DirectionProb, move, current_x, current_y)
            
            next_head = (DirectionProb + rotate) % 12
            next_state.append((next_x, next_y, next_head))
        # PSA set of three possible situation
      
        next_prob = []
        next_dic = {}
         
        for Nstate in next_state:
            nextPSA = self.psa(currentState, Nstate, policy)
           
            next_prob.append(nextPSA)
            next_dic[Nstate] = nextPSA
        
            
            
        if operation == 'get_probability':
            prob_result = random.random()
            if prob_result <= next_prob[0]:
                
                return next_state[0]
            elif next_prob[0] < prob_result <= next_prob[0] + next_prob[1]:
               
                return next_state[1]
            else:
                
                return next_state[2]
        elif operation == 'look_up':
            return next_dic
        
# Question 2 ################################################################
    # Calculate reward (state[0]=x state[1]=y)
    def rewardFunction(self, state):
        if state[0] == 0 or state[0] == self.l-1 or state[1] ==0 or state[1] == self.w - 1:
            reward = -100
        elif (state[0] == 2 or state[0] == 4) and (state[1] == 2 or state[1] == 3 or state[1] == 4):
            reward = -10
        elif state[0] == 3 and state[1] == 4:
            reward = 1
        else:
            reward = 0
        return reward
        
    ####### prepare for 5(b)##########
    def rewardFunction2(self, state):
        if state[0] == 0 or state[0] == self.l-1 or state[1] ==0 or state[1] == self.w - 1:
            reward = -100
        elif (state[0] == 2 or state[0] == 4) and (state[1] == 2 or state[1] == 3 or state[1] == 4):
            reward = -10
        elif state[0] == 3 and state[1] == 4 and state[2] in down_head:
            reward = 1
        else:
            reward = 0
        return reward


 # Question 3a  ############################################################
    # Get the move and rotate for each state
    def GetPolicy(self, current_state):
        move = 0
        rotate = 0
        x = current_state[0]
        y = current_state[1]
        head = current_state[2]
        deltaX = self.goal[0] - x
        deltaY = self.goal[1] - y

        # At goal no move & rotate
        # If front, move forward ( 1 )
        # If behind, move backward (- 1 )
        if deltaX == 0 and deltaY == 0:
            return 0, 0

        if head in up_head:
            move = 1 if (deltaY >= 0) else -1
            if deltaX == 0:
                rotate = 0
            else:
                rotate = 1 if deltaX > 0 else -1
                
        elif head in down_head:
            move = -1 if (deltaY > 0) else 1
            if deltaX == 0:
                rotate = 0
            else:
                rotate = -1 if deltaX > 0 else 1
                
        elif head in left_head:
            move = -1 if (deltaX >=0) else 1
            if deltaY == 0:
                rotate = 0
            else:
                rotate = 1 if deltaY > 0 else -1
                
        elif head in right_head:
            move = 1 if (deltaX >= 0) else -1
            if deltaY == 0:
                rotate = 0
            else:
                rotate = -1 if deltaY > 0 else 1

        return move, rotate

    # Question 3b ###########################################################
    # Plot trajactory 
    def plot_trajectory(self, current_state, policy=None):
        
        if policy is None:
            policy = {}
            for state in self.All_State_Situation:
                policy[state] = self.GetPolicy(state)
        
        state_detail = (current_state[0], current_state[1], current_state[2]% 12)
        ToDo =  policy[state_detail]
        location = (state_detail[0] + 0.5, state_detail[1] + 0.5)
       
        fig, ax = plt.subplots()
        # Build map
        ax.add_patch(patches.Rectangle((0, 0), 1, 6, color='red'))
        ax.add_patch(patches.Rectangle((1, 0), 5, 1, color='red'))
        ax.add_patch(patches.Rectangle((5, 1), 1, 5, color='red'))
        ax.add_patch(patches.Rectangle((0, 5), 5, 1, color='red'))
        ax.add_patch(patches.Rectangle((4, 2), 1, 3, color='yellow'))
        ax.add_patch(patches.Rectangle((2, 2), 1, 3, color='yellow'))
        ax.add_patch(patches.Rectangle((3, 4), 1, 1, color='green'))    
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.grid(which='major', axis='x', linewidth=0.5, linestyle='-', color='black')
        ax.grid(which='major', axis='y', linewidth=0.5, linestyle='-', color='black')
        plt.plot(location[0], location[1], 'o', markersize='20', color='y', )
        trajactory = []
        motions = []
        trajactory.append(state_detail)
        motions.append(ToDo)
        
        while ToDo != (0, 0):
           
            nextState_detail = self.nextState(state_detail, ToDo, 'get_probability')
           
            ToDo =  policy[nextState_detail]
          
            path_value = (nextState_detail[0]+0.5, nextState_detail[1]+0.5, nextState_detail[2] % 12)  
            current_loc = (state_detail[0]+0.5, state_detail[1]+0.5)
            trajactory.append(nextState_detail)
            motions.append(ToDo)
           
            plt.plot(path_value[0], path_value[1], 'o',markersize='10' )
                  
            plt.plot([current_loc[0],path_value[0]],[current_loc[1], path_value[1]],'k--')
            
           
            ax.arrow(path_value[0], path_value[1],0.5*math.sin(path_value[2]*math.pi/6),0.5*math.cos(path_value[2]*math.pi/6), 
                     head_width=0.2,head_length=0.2, fc='k', ec='k')
           
            state_detail = nextState_detail
        
        print('policy is:')
        for path in  motions:
            
            print('->', path)
            
        print('trajactory is:')
        for points in trajactory:
            
             print ('->',points)
            
        plt.show()

 
        

    # Question 3d ################################################################
    def policy_evaluation(self, discount_factor, accuracy, policy=None):
        Vk_0 = {}
        for state in self.All_State_Situation:
            Vk_0[state] = 0
        if policy is None:
            policy = {}
            for state in self.All_State_Situation:
                policy[state] = self.GetPolicy(state)
        
        delta = 1
        while delta > accuracy:
            delta = 0
            for state in self.All_State_Situation:
                Vk_1 = 0
                Policy = policy[state]
                reward = self.rewardFunction(state)
                NEXTstate = self.nextState(state, Policy, 'look_up')
                
                for nexts in NEXTstate:
                  
                    psa_s = NEXTstate[nexts]
                    
                    Vk_1 = Vk_1 + psa_s * (reward + discount_factor * Vk_0[nexts])
                delta = max(delta, abs(Vk_1 - Vk_0[state]))

                Vk_0[state] = Vk_1
            
        return Vk_0
 


  # Question 3f ################################################################
    def One_Step_Lookahead(self, Vk_0_next, discount_factor):
        one_step_policy = {}
        for state in self.All_State_Situation:
            Vk_1  = np.zeros(7)
            reward = self.rewardFunction(state)
            for x in range(7):
                Policy= Action[x]
                NEXTstate = self.nextState(state, Policy, 'look_up')
                for nexts in NEXTstate:
                    psa_s = NEXTstate[nexts]
                    Vk_1[x] += psa_s * (reward + discount_factor * Vk_0_next[nexts])
            best_index = np.argmax(Vk_1)
            one_step_policy[state] = Action[best_index]
        return one_step_policy
    
    # Question 3g ################################################################
    def policy_iteration(self, policy, discount_factor, accuracy):
        policy_choosing = 1
        while policy_choosing:
            Value = self.policy_evaluation(discount_factor, accuracy,policy)
           
            best_next_policy = self.One_Step_Lookahead(Value, discount_factor)
            if best_next_policy != policy:
                policy = best_next_policy  
            else:
                policy_choosing = 0
                print('Policy iteration done')
        return best_next_policy, Value
    
    # Question 4a ################################################################
    def value_iteration(self, discount_factor, accuracy, reward_method=None):
        value = {}
        policy = {}
        for state in self.All_State_Situation:
            policy[state] = 0
            value[state] = 0

        delta = 1
        while delta > accuracy:
            delta = 0
            for state in self.All_State_Situation:
                if reward_method is None:
                    reward = self.rewardFunction(state)
                elif reward_method == '2':
                    reward = self.rewardFunction2(state)
                    
                test_value = np.zeros(7)
                for i in range(7):
                    Policy = Action[i]
                    NEXTstate = self.nextState(state, Policy, 'look_up')
                    for nexts in NEXTstate:
                    
                        psa_s = NEXTstate[nexts]
                        test_value[i] += psa_s * (reward + discount_factor * value[nexts])
                        
                better_value = np.max(test_value)
                better_policy = Action[np.argmax(test_value)]
                delta = max(delta, abs(better_value - value[state]))
                value[state] = better_value
                policy[state] = better_policy
        print('Value iteration done')
        return policy,value


if __name__ == '__main__':
    situation1 = Envir(Width, Length, 0)
    

    
    state0 = (1, 4, 6)
    print('------3c Trajectory for initial policy------------')
    situation1.plot_trajectory(state0)
    v = situation1.policy_evaluation(0.9, 0.00001)
    state_form = (state0[0], state0[1], state0[2]%12)
    print()
   
    print('------3e Value of trajactpry at beginning------------')
    print('The value of trajactory when intial state is',state0,'is', v[state_form])
    print()
    
    
    policy0 = {}
    for x in range(Width):
        for y in range(Length):
            for z in range(12):
                policy0[(x, y, z)] = situation1.GetPolicy((x, y, z))
    start_time = time.time()
    policy_best, value_best = situation1.policy_iteration(policy0, 0.9, 0.00001)
    end_time = time.time()
    run_time = end_time - start_time
    
    print('------3h Trajectory for policy iteration------------')
    situation1.plot_trajectory(state0, policy_best)
  
    print('The value for state', state0, 'is', value_best[state_form])
    print()
    
    print('------3i Running time for policy iteration------------')
    print('Running time:', run_time)
    print()

    
    print('------4b Trajectory for value iteration------------')
    start_time = time.time()
    policy_best2,value_best2 = situation1.value_iteration(0.9, 0.00001)
    end_time = time.time()
    run_time = end_time - start_time
    situation1.plot_trajectory(state0, policy_best2)
    print('The value for state', state0, 'is', value_best2[state_form])
    print()

    print('------4c Running time for value iteration------------')
    print('Running time:', run_time)
    print()

    print('------5a Trajectory when pe = 25%------------')
    situation2 = Envir(Width, Length, 0.25)
    policy_best3,value_best3 = situation2.value_iteration(0.9, 0.00001)
    situation2 .plot_trajectory(state0, policy_best3)
    print('The value for state', state0, 'is', value_best3[state_form])
    print()

    print('------5b Trajectory with second reward function (pe = 0%)------------')
    situation3 = Envir(Width, Length, 0)
    policy_best4,value_best4 = situation3 .value_iteration(0.9, 0.00001, '2')
    situation3 .plot_trajectory(state0, policy_best4)
    print('The value for state', state0, 'is', value_best4[state_form])
    print()

    print('------5b Trajectory with second reward function(pe = 25%)------------')
    situation4 = Envir(Width, Length, 0.25)
    policy_best5,value_best5 = situation4 .value_iteration(0.9, 0.00001, '2')
    situation4 .plot_trajectory(state0, policy_best5)
