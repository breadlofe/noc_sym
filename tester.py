import numpy as np
def get_all_actions(size):
    total_actions = []
    for d in range(size-1):
        for i in range(size*d, size-1 + size*d):
            for j in range(i, size-1 + size*d):
                x1 = i
                x2 = j+1
                y1 = x1 + size
                y2 = x2 + size
                total_actions.append([x1,x2,y1,y2])
    return total_actions

def get_all_actions_better(size):
    total_actions = []
    for d in range(size-1):
        for i in range(size*d, size-1 + size*d):
            for j in range(i, size-1 + size*d):
                x1 = i
                x2 = j+1
                for q in range(1, size-d):
                    y1 = x1 + (size*q)
                    y2 = x2 + (size*q)
                    total_actions.append([x1,x2,y1,y2])
    return total_actions
    
print(get_all_actions_better(4))

hi = np.array([1.0,1.5,2.0,1.1,0.0],np.float64)
print(hi.shape)

kys = []
h = np.array(kys)
h = np.append(h, 4.0)
print(h[0])