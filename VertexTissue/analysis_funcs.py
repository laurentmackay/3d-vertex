
import numpy as np
from VertexTissue.Geometry import unit_vector
from VertexTissue.util import last_dict_value, first_dict_value

def buckle_angle_finder(G, edge=None):
    centers=set(G.graph['centers'])
    n0 = set(G.neighbors(edge[0]))
    n1 = set(G.neighbors(edge[1]))
    center=list(set.difference(set.intersection(n1, centers),set.intersection(n0, centers) ))[0]
    nhbrs = np.sort(list(G.neighbors(center)))
    nhbrs = np.array([n for n in nhbrs if n not in edge])
    lr = nhbrs[[np.sum([nn in nhbrs for  nn in G.neighbors(n)])==1 for n in nhbrs]]


    def inner(G):

        a=G.node[center]['pos']
        b=G.node[edge[1]]['pos']
        c=G.node[edge[0]]['pos']
        d=G.node[lr[0]]['pos']
        e=G.node[lr[1]]['pos']

        ab=unit_vector(a,b)
        bc=unit_vector(b,c)
        bd=unit_vector(b,d)
        be=unit_vector(b,e)
        normal = (np.cross(ab,bd)+np.cross(be,ab))/2

        bc = bc-np.dot(bc, normal)*normal
        bc /= np.linalg.norm(bc)

        return np.arccos(np.dot(ab,bc))
        
    return inner

def max_buckle_angle(d, edge = None):
    max_angle=0.0
    G=first_dict_value(d)
    buckle_angle = buckle_angle_finder(G, edge=edge)
    for G in d.values():
        max_angle=max(buckle_angle(G),max_angle)
    
    return max_angle

def final_buckle_angle(d, edge = None):
    G=last_dict_value(d)
    buckle_angle = buckle_angle_finder(G, edge=edge)
    max_angle=buckle_angle(G)
    
    return max_angle