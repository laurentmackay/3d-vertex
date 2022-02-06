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
t_intercalate = 10           # time for myosin to begin accumulating on intercalating edges
t_1 = 0                    # time for inner arc to begin 
t_2 = 10                   # time for outer arc to begin 
t_belt = 0                # time for belt to begin

# geometrical set-up
hex = 7 
pit_centers = [0,7,74,78,12,71,67]

inner_arc = [19,20,21,88,89,156,157,146,147,148,159,160,91,92,24,25,26,23,86,85,154,153,144,142,143,150,151,82,83,18]
outer_arc = [39,40,41,112,113,180,181,242,243,298,299,286,287,276,277,278,289,290,301,302,245,246,183,184,115,116,44,45,46,43,110,109,178,177,240,239,296,295,284,283,274,272,273,280,281,292,293,236,237,174,175,106,107,38]

inter_edges = [[301,302],[295,296],[292,293],[298,299],[45,46],[39,40],[272,273],[174,175],[180,181],[276,277],[183,184],[177,178],[112,113],[286,287],[289,290],[115,116],[109,110],[283,284],[280,281],[106,107]] 



# pit_centers = [294, 282, 226, 238, 306, 356, 344]

arc1 = [252, 189, 178, 109, 98, 97, 86, 85, 73, 72, 144, 142, 212, 210, 273, 272, 330, 332]
arc2 = [122, 53, 46, 45, 44, 35, 34, 25, 24, 15, 14, 5, 4, 3, 2, 8, 69, 82, 151, 162, 225, 236, 293, 292, 343, 342]
arc3 = [196, 195, 184, 183, 172, 171, 160, 159, 148, 147, 146, 157, 156, 89, 88, 21, 20, 19, 28, 95, 106, 175, 186, 249]
arc4 = [314, 313, 302, 301, 290, 289, 278, 277, 276, 287, 286, 299, 298, 243, 242, 181, 180, 113, 112, 41, 40, 39, 48, 119]




l_apical = 3.4 
l_depth = 13.6 
v_0 = ((3/2)*np.sqrt(3)*l_apical**2)*l_depth
l_intercalation = 0.1 
l_mvmt = 0.2 

# mechanical parameters
# strength of force on edges. Given in terms of myosin motors (Force=beta*myosin) 
pit_strength = 750 
belt_strength = 750 

mu_apical = 1.         # spring coefficient apical springs
mu_basal = 1.          # spring coefficient basal springs
mu_wall =  1.          # spring coefficient wall/vertical springs

myo_beta = 10e-3            # force per myosin motor

eta = 100.0                 # viscous coefficient 

press_alpha = 0.046         # area/pressure coefficient
c_ab  = 3.0                 # bending energy coefficient

basal_offset=1000

tau = 60

save_pattern="t_*.pickle"

default_edge = {'l_rest': l_apical, 'myosin':0, 'tau': np.inf}

default_ab_linker = default_edge.copy()
default_ab_linker['l_rest'] = l_depth