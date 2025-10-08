import mcts
import multiprocessing as mp
import random

def mp_helper(mcts:mcts.MCTS, shared_list:list, C:float):
    result = mcts.run(C)
    mini = result[1].index(min(result[1]))
    best_state = result[0][mini]
    shared_list.append(best_state)


if __name__ == "__main__":

    # mp setup
    num_cores = mp.cpu_count()
    print(f"Number of CPU cores: {num_cores}")
    processes = []

    # Given network
    conns = {'A':['C','G'],'B':['C','D'],'C':['F'],'D':['C'],'E':['C','D'],'F':['A','H'],'G':['E','I'],'H':['I'],'I':['H']}
    nodes = ['A','B','C','D','E','F','G','H','I']

    with mp.Manager() as manager:

        best = manager.list([])

        for i in range(num_cores):
            noc = mcts.ns.NoC(3,nodes.copy(),conns)
            tree = mcts.MCTS(noc,[3,nodes.copy(),conns],50000)
            processes.append( mp.Process( target=mp_helper, args=(tree,best,random.uniform(0.5,3.5)) ) )

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        best_of_best = None
        for b in best:
            min = 1000
            test = b.state.run_sim()
            if test[0] + test[1] < min:
                min = test
                best_of_best = b

        best_of_best.state.print_noc()
        print(best_of_best.state.run_sim())

    # best_state.state.print_noc()
# print(best_state.state.run_sim())

    



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