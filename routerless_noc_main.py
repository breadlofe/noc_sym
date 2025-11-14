import numpy as np
import routerless_noc_sym as rns
from actor_critic import Agent

def get_all_actions(size):
    '''
    Helper function that returns all possible IMR ring placements
    and direction.
    '''
    total_actions = []
    for d in range(size-1):
        for i in range(size*d, size-1 + size*d):
            for j in range(i, size-1 + size*d):
                x1 = i
                x2 = j+1
                for q in range(1, size-d):
                    y1 = x1 + (size*q)
                    y2 = x2 + (size*q)
                    total_actions.append([x1,x2,y1,y2,0])

    # Add counter clockwise
    for action in total_actions.copy():
        tmp = action.copy()
        tmp[-1] = 1
        total_actions.append(tmp)

    return total_actions

if __name__ == '__main__':
    size = 4
    env = rns.Routerless_NoC(size, {0:[1,2,4],1:[0,2,5,6],2:[1,3,6,7],3:[2,7],4:[0,1,5],
                                 5:[1,4,6,9],6:[2,5,7,10],7:[3,6,11],8:[4,9,12],9:[5,8,10,13],
                                 10:[6,9,11,14],11:[7,10,15],12:[8,9,13],13:[9,12,14],14:[10,13,15],15:[10,11,14]})
    possible_actions = get_all_actions(size)
    agent = Agent( alpha=1e-5, n_actions=len(possible_actions))
    n_games = 300

    #filename = 'routerless.png'
    best_score = -np.inf
    best_env = None

    score_history = []

    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        #print(observation)
        while not done:
            action = agent.choose_action(observation)
            # if action == 0:
            #     print("HGKNOSGJSEDGJIOSJIOG")
            observation_, reward, done, trunc, info = env.connect_and_wire_imr(possible_actions[action-1]) #This should set up a wire with given action
            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if score > best_score:
            best_score = score
            best_env = env
            if not load_checkpoint:
                agent.save_models

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    print(env.is_terminal())
    env.run_sim()
    print(env.get_hop_count())
    print(env.imrs)
    env.print_imr()