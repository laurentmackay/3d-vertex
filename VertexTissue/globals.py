##########
#
# globals.py
#
#
# Author: Clinton H. Durney
# Email: clinton.durney@jic.ac.uk 
#
# Last Edit: 8/26/21
##########
import numpy as np

# time
t_final = 2000              # entire time of simulation
dt = 0.5                    # time step
t_pit = 0                   # time for myosin build-up (myo accumulates for t<t_pit) 
t_intercalate = 375          # time for myosin to begin accumulating on intercalating edges
t_1 = 375                # time for inner arc to begin 
t_2 = 375                   # time for outer arc to begin 
t_belt = 3500                # time for belt to begin

# geometrical set-up
hex = 7 
pit_centers = [0,7,74,78,12,71,67]

inner_arc = [19,20,21,88,89,156,157,146,147,148,159,160,91,92,24,25,26,23,86,85,154,153,144,142,143,150,151,82,83,18]
outer_arc = [39,40,41,112,113,180,181,242,243,298,299,286,287,276,277,278,289,290,301,302,245,246,183,184,115,116,44,45,46,43,110,109,178,177,240,239,296,295,284,283,274,272,273,280,281,292,293,236,237,174,175,106,107,38]

inter_edges_middle = [[301,302],[295,296],[292,293],[298,299],[45,46],[39,40],[272,273],[174,175],[180,181],[276,277],[183,184],[177,178],[112,113],[286,287],[289,290],[115,116],[109,110],[283,284],[280,281],[106,107]] 

# inter_edges = [[234,233],]
inter_edges_middle = [[234,233], [36, 35], [224, 225], [29, 30]] #4
inter_edges_middle.extend([[231,230],[228, 227], ]) #6
inter_edges_middle_bis = inter_edges_middle.copy()

inter_edges_middle.extend([[104, 103], [94, 95]]) #8
inter_edges_middle.extend([[101, 168], [166, 97], [219, 221],  [210, 212]]) #10-12 

inter_edges_middle_bis.extend([ [95, 28], [104, 34],  [168, 169],  [222, 221], [210,211],  [165, 166]]) #12-again
inter_edges_middle_bis.extend([[162, 163], [172, 171],  [31, 100], [98, 33], [214, 215], [217, 218] ]) #18

inter_edges_outer = [[56, 55], [363, 364], [49, 50], [354, 355]] #4
inter_edges_outer.extend([[360, 361], [357, 358]])#6

inter_edges_outer_bis = inter_edges_outer.copy()
inter_edges_outer.extend([[196,195], [186, 187]]) #8
inter_edges_outer.extend([[339, 337], [328, 330], [252, 189], [193, 254]])#10-12


inter_edges_outer_bis.extend([[128, 127],  [351, 340], [255, 310],  [118, 119], [342, 329], [308, 251]])#12-again
inter_edges_outer_bis.extend([[305, 248], [257, 314], [335, 349], [333, 345],  [124, 125], [121, 122]])#18

# pit_centers = [294, 282, 226, 238, 306, 356, 344]

arc1 = [252, 189, 178, 109, 98, 97, 86, 85, 73, 72, 144, 142, 212, 210, 273, 272, 330, 332]
arc2 = [122, 53, 46, 45, 44, 35, 34, 25, 24, 15, 14, 5, 4, 3, 2, 8, 69, 82, 151, 162, 225, 236, 293, 292, 343, 342]
arc3 = [196, 195, 184, 183, 172, 171, 160, 159, 148, 147, 146, 157, 156, 89, 88, 21, 20, 19, 28, 95, 106, 175, 186, 249]
arc4 = [314, 313, 302, 301, 290, 289, 278, 277, 276, 287, 286, 299, 298, 243, 242, 181, 180, 113, 112, 41, 40, 39, 48, 119]

arc1 = [181, 180, 113, 112, 41, 40, 39, 38, 107, 106, 175, 174]
arc2 = [95, 94, 163, 162, 225, 224, 211, 210, 212, 214]
arc3 = [109, 178, 177, 240, 239, 296, 295, 284, 283, 274, 272, 273]
arc4 = [276, 277, 278, 289, 290, 301, 302, 245, 246, 183, 184, 115]
arc5 = [172, 103, 104, 34, 35, 36, 33, 98, 97, 166]
arc6 = [221, 219, 218, 217, 231, 230, 169, 168, 101,100]



l_apical = 3.4 
l_depth = 13.6 
A_0=((3/2)*np.sqrt(3)*l_apical**2)
v_0 = A_0*l_depth

l_intercalation = .1
l_mvmt = l_intercalation/2

# mechanical parameters
# strength of force on edges. Given in terms of myosin motors (Force=beta*myosin) 
pit_strength = 300

belt_strength = 750 
intercalation_strength=1250

mu_apical = 1.         # spring coefficient apical springs
mu_basal = mu_apical          # spring coefficient basal springs
mu_wall =  mu_apical         # spring coefficient wall/vertical springs

myo_beta = 10e-3            # force per myosin motor

eta = 100.0                 # viscous coefficient 

press_alpha = 0.00735   # area/pressure coefficient
c_ab  = 3.0                 # bending energy coefficient



tau = 60

kc=0.004

softened=[]

save_pattern="t_*.pickle"

default_edge = {'l_rest': l_apical, 'myosin':0, 'tau': 60}

default_ab_linker = default_edge.copy()
default_ab_linker['l_rest'] = l_depth