import math
import numpy as np
from copy import copy, deepcopy
from dataclasses import dataclass

# {A: [B,C]}

@dataclass
class Wire:
    tag: str
    used: int

def wire_search(wires:dict, src:str, thing:str):
    the_list = wires[src]
    idx = 0
    for dest in the_list:
        if dest[0] == thing:
            return idx
        idx += 1
    return idx

class NoC:
    def __init__(self, size: int, nodes: list, comms: dict):
        self.size = size
        self.board = [[None]]*size*size
        self.unplaced = nodes
        self.dests = comms

    def legal_move(self, pos:int, node:str):
        if self.board[pos] == [None]:
            if node not in self.board:
                if node in self.unplaced:
                    return True
        return False
    
    def get_valid_moves(self):
        empty_slots = [i for i, x in enumerate(self.board) if x == [None]]
        return [empty_slots, self.unplaced]

    def place_node(self, pos:int, node:str):
        if self.legal_move(pos, node):
            self.board[pos] = node
            self.unplaced.remove(node)
        return self.copy()

    def wire_helper(self, links:dict, me:int, you:int, wire:Wire):
        if me in links.keys():
            links[me].append( [you, wire ] )
        else:
            links[me] = [[you, wire]]

    def wire(self):
        # {A: [B, Wire('AB',0)],[C, Wire('AC',0)]} <--- links
        # {src: dest, packets crossing this wire}
        #---
        # {A: [B,C]} <--- self.dests
        # {src: dests}
        links = dict()
        for i in range( len(self.board) ):
            if (i+1)%self.size != 0: # connect horizontal wires
                wire = Wire(self.board[i]+self.board[i+1],0)
                self.wire_helper(links,self.board[i],self.board[i+1],wire)
                self.wire_helper(links,self.board[i+1],self.board[i],wire)
                #links[self.board[i]] = [[self.board[i+1], wire]]
                #links[self.board[i+1]] = [[self.board[i], wire]]
            if int(i/self.size) != self.size-1:
                wire = Wire(self.board[i]+self.board[i+self.size],0)
                self.wire_helper(links,self.board[i],self.board[i+self.size],wire)
                self.wire_helper(links,self.board[i+self.size],self.board[i],wire)
                # if self.board[i] in links.keys():
                #     links[self.board[i]].append( [self.board[ i+self.size ], wire ] )
                # else:
                #     links[self.board[i]] = [ [ self.board[ i+self.size ], wire ] ]
                # if self.board[i+self.size] in links.keys():
                #     links[self.board[i+self.size]].append( [self.board[ i ], wire ] )
                # else:
                #     links[self.board[i+self.size]] = [ [ self.board[ i ], wire ] ]
        return links
    

    def routing_algo(self, links:dict):
        for src in self.dests:
            src_pos_x, src_pos_y = self.board.index(src)%self.size, int(self.board.index(src)/self.size)
            for dest in self.dests[src]:
                cur_x, cur_y = src_pos_x, src_pos_y
                dest_pos_x, dest_pos_y = self.board.index(dest)%self.size, int(self.board.index(dest)/self.size)
                while (cur_x,cur_y) != (dest_pos_x,dest_pos_y): 
                    if dest_pos_x > cur_x: # Go right!
                        cur_x += 1
                        links[ self.board[cur_x+(cur_y*self.size)] ][wire_search(links,self.board[cur_x+(cur_y*self.size)],self.board[cur_x+(cur_y*self.size)-1])][1].used += 1 
                    elif dest_pos_x < cur_x: # Go left!
                        cur_x -= 1
                        links[ self.board[cur_x+(cur_y*self.size)] ][wire_search(links,self.board[cur_x+(cur_y*self.size)],self.board[cur_x+(cur_y*self.size)+1])][1].used += 1 
                    elif dest_pos_y > cur_y: # Go down!
                        cur_y += 1
                        test = wire_search(links,self.board[cur_x+(cur_y*self.size)],self.board[cur_x+(cur_y*self.size)-1])
                        links[ self.board[cur_x+(cur_y*self.size)] ][wire_search(links,self.board[cur_x+(cur_y*self.size)],self.board[cur_x+((cur_y-1)*self.size)])][1].used += 1 
                    elif dest_pos_y < cur_y: # Go up!
                        cur_y -= 1
                        links[ self.board[cur_x+(cur_y*self.size)] ][wire_search(links,self.board[cur_x+(cur_y*self.size)],self.board[cur_x+((cur_y+1)*self.size)])][1].used += 1 
        return links
    

    def get_throughput(self):
        links = self.wire()
        links = self.routing_algo(links)
        # for key in links, for path in links, add .used up to total and divide by size*size
        total_usage = 0
        for keys in links:
            for paths in links[keys]:
                total_usage += paths[1].used
        return total_usage / ((self.size-1)* self.size * 2)


    def get_hop_count_single(self, src, dest):
        x_dist = abs( (self.board.index(src)%self.size) - (self.board.index(dest)%self.size) )
        y_dist = abs( int(self.board.index(src) / self.size) - int(self.board.index(dest) / self.size) )
        return x_dist + y_dist

    def get_hop_count(self):
        hc_array = []
        for src in self.dests:
            for dest in self.dests[src]:
                hc_array.append( self.get_hop_count_single(src, dest) )
        return float( np.average( np.array(hc_array) ) )
    
    def print_noc(self):
        for i in range( self.size ):
            print(self.board[i*self.size:self.size+i*self.size])

    def is_terminal(self):
        if [None] in self.board:
            return False
        return True

    def run_sim(self):
        #self.print_noc()
        hc = self.get_hop_count()
        tp = self.get_throughput()
        return (hc,tp)
    
    def copy(self):
        new = NoC(self.size, self.unplaced.copy(), self.dests)
        new.board = self.board.copy()
        return new
    
    def reward(self):
        results = self.run_sim()
        return -(results[0]**2) - results[1]

conns = {'A':['C','G'],'B':['C','D'],'C':['F'],'D':['C'],'E':['C','D'],'F':['A','H'],'G':['E','I'],'H':['I'],'I':['H']}
nodes = ['A','C','D','F','E','B','G','H','I']
noc = NoC(3,nodes.copy(),conns)
noc2 = NoC(3,nodes.copy(),conns)
noc3 = NoC(3,nodes.copy(),conns)

nodes2 = ['A','B','C','D','E','F','G','H','I']
noc2 = NoC(3,nodes2.copy(),conns)
# ACTIONS TAKEN RANDOMLY... NO THOUGHT
idx = 0
for n in nodes:
    noc.place_node(idx,n)
    noc2.place_node(idx,nodes2[idx])
    idx += 1

print(noc.run_sim())
print(noc.reward())

print(noc2.run_sim())
print(noc2.reward())

# noc2.place_node(0,'A')
# noc3 = noc2.copy()
# noc3.place_node(1,'B')
# noc2.print_noc()
# noc3.print_noc()