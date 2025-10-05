import mcts

# Given network
conns = {'A':['C','G'],'B':['C','D'],'C':['F'],'D':['C'],'E':['C','D'],'F':['A','H'],'G':['E','I'],'H':['I'],'I':['H']}
nodes = ['A','B','C','D','E','F','G','H','I']

# Example
noc = mcts.ns.NoC(3,nodes.copy(),conns)

pray = mcts.MCTS(noc,[3,nodes.copy(),conns],5000)
please = pray.run()


mini = please[1].index(min(please[1]))
best_state = please[0][mini]
#print(";;;;;;;;;")
#print(max(please[1]))
#print(min(please[1]))

print(" ======= RESULT ========")
best_state.state.print_noc()
print(best_state.state.run_sim())

"""
 ======= BEST RESULT UCB C = 2, -(hc**2) - lu ========
['B', 'D', 'E']
['A', 'C', 'G']
['F', 'H', 'I']
(1.2857142857142858, 3.0)

===== BEST RESULT WITH UCB C = 2, -2*hc - lu =====
['G', 'D', 'B']
['F', 'E', 'C']
['A', 'I', 'H']
(1.7142857142857142, 4.0)

COMPARED TO RANDOM

['A', 'B', 'C']
['D', 'E', 'F']
['G', 'H', 'I']
(1.7857142857142858, 4.166666666666667)
"""