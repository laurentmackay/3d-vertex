import os
import numpy as np
import matplotlib.pyplot as plt
from VertexTissue.Events import EventListenerPair

from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import T1_minimal, tissue_3d, get_outer_belt
import VertexTissue.SG as SG
import VertexTissue.T1 as T1

from VertexTissue.Analysis import parameter_sweep, sweep, parameter_sweep_analyzer

from VertexTissue.globals import default_ab_linker, default_edge
import VertexTissue.globals as const


from VertexTissue.util import last_dict_value, script_name, signature_string
from VertexTissue.Geometry import euclidean_distance
from VertexTissue.visco_funcs import edge_crumpler


try:
    from VertexTissue.PyQtViz import edge_view
    import matplotlib.pyplot as plt
    viewable=True
    base_path = './data/'
except:
    viewable=False
    base_path = '/scratch/st-jjfeng-1/lmackay/data/'


def final_length(d):
    G = last_dict_value(d)
    # edge_view(G)
    a = G.nodes[1]['pos']
    b = G.nodes[11]['pos']
    return euclidean_distance(a,b)
edges=((1,11),(10,11))

def shortest_length(d):
    lens = []
    for G in d.values():
        lens.append(min(*[euclidean_distance(G.nodes[e[0]]['pos'],G.nodes[e[1]]['pos']) for e in edges]))

    # edge_view(G)

    return min(lens)


def l_ss(phi, ec=0.2, phi0=1.0):
    if phi<ec:
        return 1-phi
    elif phi<=phi0:
        return (1-ec)+(1-ec)*(ec-phi)/(phi0-ec)
    else:
        return 0.0

def plot_results(forces,results):

    inter_len = const.l_intercalation/const.l_apical
    
    lens = results/const.l_apical


    phi = forces*const.myo_beta/const.l_apical
    phi_fine = np.linspace(phi[0],phi[-1],200)

    plt.plot(phi, lens[:,0],'-k',label='elastic ($\phi_0=1$)')

    plt.plot(phi, lens[:,1],'-r',label='$\phi_0=0.3$  (3D)')
    plt.plot(phi, lens[:,4],'--r')

    plt.plot(phi, lens[:,2],'-g',label='$\phi_0=0.6$  (3D)')
    plt.plot(phi, lens[:,5],'--g')

    plt.plot(phi, lens[:,3],'-b',label='$\phi_0=0.8$  (3D)')
    plt.plot(phi, lens[:,6],'--b')

    

    plt.plot([phi[0], phi[-1]],[inter_len, inter_len],':k')


    plt.xlabel('$-\phi$ (peripheral)', fontsize=14)
    plt.ylabel('Shortest Edge Length (norm.)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    # plt.savefig('T1_cable_VE.pdf',dpi=200)
    plt.show()


def run(force, visco=False, phi0=1.0, ndim=3):

    #
    G, G_apical = tissue_3d( gen_centers=T1_minimal,  basal=(ndim==3))
    
    belt = get_outer_belt(G_apical)


    squeeze = SG.just_belt(G, belt, t=0, strength=force)

   
    if not visco:
        kw={}
    else:
        kw={'rest_length_func': edge_crumpler(G, phi0=phi0)}


    terminate_simulation, listen_for_intercalation = EventListenerPair()
    
    const.l_intercalation=0.1
    const.l_mvmt=0.001

    if ndim==2:
        const.press_alpha*=4
    #create integrator
    integrate = monolayer_integrator(G, G_apical, 
                                    pre_callback=squeeze, 
                                    intercalation_callback=terminate_simulation, 
                                    termination_callback=listen_for_intercalation,  
                                    blacklist=True,
                                    player=False, viewer={'button_callback':terminate_simulation} if viewable else False, minimal=False,  ndim=ndim, **kw)
    #integrates
    # try:
    print(f'starting f={force}')



    pattern=os.path.join(base_path, script_name(), signature_string()+'.pickle')

    integrate(5, 3000, 
            dt_init = 1e-3,
            adaptive=True,
            dt_min=1e-2,
            save_rate=5,
            save_pattern=pattern)





phi0s=[0.3, 0.6, .8]

if __name__ == '__main__':
    forces=np.linspace(0, 1200, 4)

    kws=[{},#base function
         {'ndim':2}, #2D
         *[{'visco':True, 'phi0':phi0} for phi0 in phi0s], #viscoelastic
         *[{'visco':True, 'phi0':phi0, 'ndim':2} for phi0 in phi0s] #viscoelastic 2D
         ]



    sweep(forces, run, kw=kws, savepath_prefix=base_path+script_name(), overwrite=False)


        
    # run(0.00001, visco=True, phi0=0.3, ndim=3)
    # 
    print('done')
