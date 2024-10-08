##########
#
# globals.py
#
#
# Author: Clinton H. Durney
# Email: cdurney@math.ubc.ca
#
# Last Edit: 6/14/19
##########
import numpy as np

# time
t_final = 20000         # entire time of simulation
dt = 0.5               # time step
t_pit = 0             # time for myosin build-up (myo accumulates for t<t_pit) 
t_intercalate = 375    # time for myosin to begin accumulating on intercalating edges
t_inner = 375              # time for inner arc to saturate 
t_outer = 375          # time for outer arc to begin 
t_2 = 2754   #2441 for BE=1           # time for outer arc to saturate 
t_belt = 3500          # time for belt to begin
t_3 = t_belt+2700              # time for AC belt to saturate -- circumferential belt

# geometrical set-up
hex = 7 
pit_centers = [0,7,74,78,12,71,67]

inner_arc = [19,20,21,88,89,156,157,146,147,148,159,160,91,92,24,25,26,23,86,85,154,153,144,142,143,150,151,82,83,18]
outer_arc = [39,40,41,112,113,180,181,242,243,298,299,286,287,276,277,278,289,290,301,302,245,246,183,184,115,116,44,45,46,43,110,109,178,177,240,239,296,295,284,283,274,272,273,280,281,292,293,236,237,174,175,106,107,38]

l_apical = 3.4 
l_depth = 13.6 
v_0 = ((3/2)*np.sqrt(3)*l_apical**2)*l_depth
# pit_radius = 25 
# l_tol = 10e-1
l_intercalation = 0.1 
l_mvmt = 0.2 

# mechanical parameters
pit_strength = 540
belt_strength = 750
intercalation_strength = 1250

mu_apical = 1.         # spring coefficient apical springs
mu_basal = 1.          # spring coefficient basal springs
mu_wall =  1.          # spring coefficient wall/vertical springs

myo_beta = 10e-3            # force per myosin motor

eta = 100.0                 # viscous coefficient 

press_alpha = 0.046         # area/pressure coefficient
c_ab  = 3.0                 # bending energy coefficient

