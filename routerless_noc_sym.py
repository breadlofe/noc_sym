import math
import numpy as np
from copy import copy, deepcopy
from dataclasses import dataclass

# {A: [B,C]}

@dataclass
class IMR:
    tag: str
    dim: list
    dir: int
    used: int

def place_nodes_on_board(size:int):
    '''
    Helper function for placing values on NoC board.
    '''
    out = [[None]]*size*size
    for i in range(size*size):
        out[i] = i
    return out

def in_column(src:int, dest:int, size:int):
    '''
    Helper function to find if a node is in the same column as another.
    '''
    tmp = src
    while tmp < size*size:
        if tmp + size == dest:
            return True
        tmp = tmp + size
    return False

class Routerless_NoC:
    def __init__(self, size: int, n_con: dict):
        self.size = size
        self.board = place_nodes_on_board(size)
        self.n_con = n_con #neuron connection that guide what node connects to what in NoC (what packets are sent where)
        self.imrs = []
        self.imr_paths = [] #this is the state
        self.imr_codified = []

    def reset(self):
        self.imrs = []
        self.imr_paths = []
        self.imr_codified = []

        return np.array([0.0,1.0,4.0,5.0,0.0],np.float64)

    def wire_imr(self, dim:list):
        '''
        Setup the IMR Ring for the NoC.
        '''
        # dataclass with tag, dimensions of wire, direction, and packets traveled.
        # [x1,x2,y1,y2]
        # CW (0) or CCW (1)

        # === ERROR HANDLING ===
        if dim[0] >= dim[1] or dim[2] >= dim[3]:
            raise ValueError("Dimensions given to not line up row-wise. Must be rect.")
        if not in_column(dim[0], dim[2], self.size) or not in_column(dim[1], dim[3], self.size):
            raise ValueError("Dimensions given to not line up column-wise. Must be rect.")
        dir = dim[-1] % 2
        # ======================
        
        new_imr = IMR(len(self.imrs), dim[:4], dir, 0)
        # new_imr.dim = dim
        # new_imr.dir = dir
        # new_imr.used = 0
        # new_imr.tag = len(self.imrs)
        self.imrs.append(new_imr)

        # observation_, reward, done, trunc, info
        return self.imr_paths.copy(), self.reward(), self.is_terminal(), 0, 0
    
    def connect_and_wire_imr(self, dim:list):
        '''
        Setup the IMR Ring for the NoC.
        '''
        # dataclass with tag, dimensions of wire, direction, and packets traveled.
        # [x1,x2,y1,y2]
        # CW (0) or CCW (1)

        # === ERROR HANDLING ===
        if dim[0] >= dim[1] or dim[2] >= dim[3]:
            raise ValueError("Dimensions given to not line up row-wise. Must be rect.")
        if not in_column(dim[0], dim[2], self.size) or not in_column(dim[1], dim[3], self.size):
            raise ValueError("Dimensions given to not line up column-wise. Must be rect.")
        dir = dim[-1] % 2
        # ======================
        
        new_imr = IMR(len(self.imrs), dim[:4], dir, 0)
        # new_imr.dim = dim
        # new_imr.dir = dir
        # new_imr.used = 0
        # new_imr.tag = len(self.imrs)
        self.imrs.append(new_imr)

        # CONNECT THE PATH
        cur_path = []
        # For clockwise
        if new_imr.dir == 0:
            # for i in range from x1 to x2, add those nodes to thing 
            for i in range(new_imr.dim[0], new_imr.dim[1]):
                cur_path.append(self.board[i]) 
            # for i in range from x2 to y2 add those nodes to thing
            for i in range(new_imr.dim[1], new_imr.dim[3], self.size):
                cur_path.append(self.board[i]) 
            # for i in range from y2 to y1 add those nodes to thing
            for i in range(new_imr.dim[3], new_imr.dim[2], -1):
                cur_path.append(self.board[i]) 
            # for i in range from y1 to x1 add those nodes to thing
            for i in range(new_imr.dim[2], new_imr.dim[0], -self.size):
                cur_path.append(self.board[i]) 
        # For counter-clockwise
        else:
            # for i in range from x1 to y1, add those nodes to thing 
            for i in range(new_imr.dim[0], new_imr.dim[2], self.size):
                cur_path.append(self.board[i]) 
            # for i in range from y1 to y2 add those nodes to thing
            for i in range(new_imr.dim[2], new_imr.dim[3]):
                cur_path.append(self.board[i]) 
            # for i in range from y2 to x2 add those nodes to thing
            for i in range(new_imr.dim[3], new_imr.dim[1], -self.size):
                cur_path.append(self.board[i]) 
            # for i in range from x2 to x1 add those nodes to thing
            for i in range(new_imr.dim[1], new_imr.dim[0], -1):
                cur_path.append(self.board[i]) 

        self.imr_paths.append(cur_path)
        self.imr_codified = np.array(dim,dtype=np.float64)
        # observation_, reward, done, trunc, info
        return np.array(self.imr_codified.copy(),dtype=np.float64), self.reward(), self.is_terminal(), 0, 0
    
    def reward(self):
        if not self.is_terminal():
            return 0
        else:
            self.run_sim()
            return max(1000-self.get_hop_count(),0)

    def create_path(self):
        '''
        Create the path of nodes that will be seen on a specific IMR ring.
        '''
        nodes_on_imr_path = []
        for imr in self.imrs:
            cur_path = []
            # For clockwise
            if imr.dir == 0:
                # for i in range from x1 to x2, add those nodes to thing 
                for i in range(imr.dim[0], imr.dim[1]):
                    cur_path.append(self.board[i]) 
                # for i in range from x2 to y2 add those nodes to thing
                for i in range(imr.dim[1], imr.dim[3], self.size):
                    cur_path.append(self.board[i]) 
                # for i in range from y2 to y1 add those nodes to thing
                for i in range(imr.dim[3], imr.dim[2], -1):
                    cur_path.append(self.board[i]) 
                # for i in range from y1 to x1 add those nodes to thing
                for i in range(imr.dim[2], imr.dim[0], -self.size):
                    cur_path.append(self.board[i]) 
            # For counter-clockwise
            else:
                # for i in range from x1 to y1, add those nodes to thing 
                for i in range(imr.dim[0], imr.dim[2], self.size):
                    cur_path.append(self.board[i]) 
                # for i in range from y1 to y2 add those nodes to thing
                for i in range(imr.dim[2], imr.dim[3]):
                    cur_path.append(self.board[i]) 
                # for i in range from y2 to x2 add those nodes to thing
                for i in range(imr.dim[3], imr.dim[1], -self.size):
                    cur_path.append(self.board[i]) 
                # for i in range from x2 to x1 add those nodes to thing
                for i in range(imr.dim[1], imr.dim[0], -1):
                    cur_path.append(self.board[i]) 

            nodes_on_imr_path.append(cur_path)
        
        self.imr_paths = nodes_on_imr_path
        return nodes_on_imr_path
    
    def send_packet(self, src, dest, verbose=False):
        '''
        Send the packet around the IMR ring to dest. Will inject onto path with
        smallest route time to destination.
        '''
        cur_path = None
        cur_idx = None
        best_dist = 1000000000
        for path in self.imr_paths:
            if src in path and dest in path:
                tmp_dist = (path.index(dest) - path.index(src)) % len(path)
                if tmp_dist < best_dist:
                    cur_path = path
                    cur_path_idx = self.imr_paths.index(path)
                    best_dist = tmp_dist
                #break

        # Error handling stuff
        if cur_path == None:
            raise ValueError("Nodes not connected!")
        
        # send it around loop!
        src_idx = cur_path.index(src)
        cur_idx = src_idx
        dest_idx = cur_path.index(dest)
        if verbose:
            print(f"\nSending packet from {cur_path[src_idx]} to {cur_path[dest_idx]} on IMR {cur_path_idx}")
        while cur_idx != dest_idx:
            if verbose:
                print(f"Packet at {cur_path[cur_idx]}")
            self.imrs[cur_path_idx].used += 1
            cur_idx = (cur_idx + 1) % len(cur_path)
        if verbose:
            print(f"Packet received at {cur_path[cur_idx]}!\n")

    def get_hop_count(self):
        '''
        Get hop count for routerless noc by looking at IMR usage.
        '''
        total = 0
        for imr in self.imrs:
            total += imr.used
        return total
    
    def print_noc(self):
        for i in range( self.size ):
            print(self.board[i*self.size:self.size+i*self.size])

    def print_imr(self):
        idx = 0
        for imr in self.imr_paths:
            tmp_board = self.board.copy()
            for node in imr:
                tmp_board[tmp_board.index(node)] = '+' if node not in self.imrs[idx].dim else '*'
            idx += 1
            print()
            for i in range( self.size ):
                print(tmp_board[i*self.size:self.size+i*self.size])

    def is_terminal(self):
        '''
        Can each neuron speak to the node that it wants to?
        Key is src and its list in dict is the destinations.

        '''
        try:
            for key in self.n_con:
                for node in self.n_con[key]:
                    self.send_packet(key, node, False)
            for imr in self.imrs:
                imr.used = 0
            return True

        except:
            return False

    def run_sim(self, verbose=False):
        '''
        Send a bunch of packets to test the network, yo!
        '''
        for key in self.n_con:
            for node in self.n_con[key]:
                self.send_packet(key, node, verbose)

        return self.get_hop_count()

def test_me():
    test = Routerless_NoC(4, {0:[1,2,4],1:[0,2,5,6],2:[1,3,6,7],3:[2,7],4:[0,1],5:[1,4],6:[7,10]})
    test.wire_imr([0,3,4,7,0])
    print(test.is_terminal())
    test.wire_imr([6,7,10,11,1])
    print(test.create_path())
    # test.send_packet(6,11,True)
    # test.send_packet(10,6,True)
    # test.print_noc()
    # test.print_imr()
    # print(test.get_hop_count())
    print(test.run_sim(verbose=True))
    print(test.is_terminal())
    #test.send_packet(6,7,True)
    # noc2.place_node(0,'A')
    # noc3 = noc2.copy()
    # noc3.place_node(1,'B')
    # noc2.print_noc()
    # noc3.print_noc()