# Explanation

This is a proof of concept project for evaluation the effectiveness of using MCTS to plot nodes on a NxN NoC mesh topology. Evaluated metrics are hop count and link usage score based on how balanced the load of the NoC is. 

## Usage

```python
import mcts

# Given network parameters (what nodes are sending packets to what, and the titles of those nodes).
conns = {'A':['C','G'],'B':['C','D'],'C':['F'],'D':['C'],'E':['C','D'],'F':['A','H'],'G':['E','I'],'H':['I'],'I':['H']}
nodes = ['A','B','C','D','E','F','G','H','I']

# Initialization of NoC.
noc = mcts.ns.NoC(3,nodes.copy(),conns)

# Run MCTS algorithm and get output.
tree = mcts.MCTS(noc,[3,nodes.copy(),conns],5000)
results = tree.run()

# Show results in meaningful way.
min_idx = results[1].index(min(results[1]))
best_state = results[0][min_idx]

print(" ======= RESULT ========")
best_state.state.print_noc()
print(best_state.state.run_sim())
```
## Thanks To
John Levine for a very helpful breakdown of the MCTS algortihm: https://www.youtube.com/watch?v=UXW2yZndl7U
Andrew Barto and Richard S. Sutton for their work on Reinforcement Learning: An Introduction