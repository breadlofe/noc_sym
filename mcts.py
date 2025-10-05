# Dustin Simpkins
# Special thanks to John Levine: https://www.youtube.com/watch?v=UXW2yZndl7U
# Also special thanks to the authors of Reinforcement Learning: An Introduction
import math
import numpy as np
import noc_sym as ns
import copy
#import matplotlib.pyplot as plt

def ucb_score(parent, child):
    '''
    Score function of the MCTS used by best child selection.
    Look at the visit counts of the parent and child, multiplied
    by some constant C (C = 2 here, for some incentivized exploration).
    '''
    prior_score = 2 * math.sqrt( math.log(parent.visit_count) ) / ( child.visit_count + 1 )
    if child.visit_count > 0:
        value_score = child.value()
    else:
        value_score = 0
    return value_score + prior_score

class Node:
    def __init__(self, state: ns.NoC):
        self.visit_count = 0
        self.value_sum = 0
        self.children = []
        self.state = state

    def expanded(self):
        '''
        If the node has children/has been expanded.
        '''
        return len(self.children) > 0
    
    def value(self):
        '''
        The cumulative average value of the node in the tree.
        '''
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def best_child(self):
        '''
        Select best child best on UCB score of that child.
        '''
        best_score = -np.inf
        best_child = None

        for child in self.children:
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child
    
    def copy(self):
        new_node = Node(self.state.copy())
        new_node.children = self.children.copy()
        new_node.value_sum = self.value_sum
        new_node.visit_count = self.visit_count
        return new_node
    
    def expand(self):
        '''
        Get all the combinations of possible (and legal) board configurations 
        from the given state. Add these to the children of this node.
        '''
        valid_moves = self.state.get_valid_moves()
        for empty_slots in valid_moves[0]:
            for unplaced_nodes in valid_moves[1]:
                new_state = self.state.copy()
                new_state.place_node(empty_slots, unplaced_nodes)
                self.children.append( Node(new_state) )


class MCTS:
    # params --> size, nodes, communications
    def __init__(self,env,env_params,simulations):
        self.env = None
        self.env_params= env_params
        self.simulations = simulations

    def rollout(self, node:Node):
        '''
        Rollut functionality of MCTS: take random actions until you hit
        a terminal state. Then, return the value of that terminal state.
        '''
        # ROLLOUT PHASE
        if not node.state.is_terminal():
            while not node.state.is_terminal():
                next_action = [ int(np.random.choice( node.state.get_valid_moves()[0]) ), str(np.random.choice( node.state.get_valid_moves()[1] )) ]
                next_state = node.state.place_node(next_action[0], next_action[1]) # next_state

            # The value of the new state
            value = next_state.reward()
        else:
            value = node.state.reward()
        return value

    def run(self):
        '''
        The MCTS algorithm. Run simulations and return back optimal terminal states
        and their values.
        '''
        # env is NoC, state is the board, action is placing node onto board
        self.env = ns.NoC(self.env_params[0], self.env_params[1].copy(), self.env_params[2])
        state = self.env.copy()
        root = Node(state.copy())
        optimal_states = []
        optimal_values = []
        for _ in range(self.simulations):
            current = root
            search_path = [current]

            # helper
            if _ % 200 == 0 and _ != 0:
                print(f"{_} simulations ran...")

            # is current a leaf node? IF IT IS EXPANDED, THEN IT IS NOT A LEAF NODE !!!!
            while current.expanded(): # NO!
                current = current.best_child()
                search_path.append(current)
            # YES!
            # is the visits for the current node 0?
            if current.visit_count == 0: # YES
                value = self.rollout(current.copy()) 
                self.backpropagate(search_path, value)
            else:   # NO!
                if not current.state.is_terminal():
                    current.expand()
                    current = current.children[0]
                    search_path.append(current)
                    value = self.rollout(current.copy())
                    self.backpropagate(search_path, value)
            
            # Tracking
            if current.state.is_terminal():
                optimal_states.append(current)
                optimal_values.append(current.value_sum)

        return optimal_states, optimal_values
    
        
    def backpropagate(self, search_path, value):
        '''
        At the end of a simulation, we propogate the evaluation all the way up the tree 
        to the root.
        '''
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1


# conns = {'A':['C','G'],'B':['C','D'],'C':['F'],'D':['C'],'E':['C','D'],'F':['A','H'],'G':['E','I'],'H':['I'],'I':['H']}
# nodes = ['A','B','C','D','E','F','G','H','I']

# noc = ns.NoC(3,nodes.copy(),conns)

# pray = MCTS(noc,[3,nodes.copy(),conns],500000)
# please = pray.run()


# mini = please[1].index(min(please[1]))
# best_state = please[0][mini]
# print(";;;;;;;;;")
# print(max(please[1]))
# print(min(please[1]))

# print(" ======= RESULT ========")
# best_state.state.print_noc()
# print(best_state.state.run_sim())

# plt.title( f"Average reward over {len(please[1])} terminals hit" )
# plt.plot(np.arange(len(please[1])), please[1])
# plt.xlabel("Number of t hits")
# plt.ylabel("Average reward percentage")
# plt.show()