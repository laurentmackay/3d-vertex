import copy
import os

from matplotlib import colors

import VertexTissue
from VertexTissue.Iterable import first_item
from VertexTissue.Memoization import function_call_savepath
from VertexTissue.TissueForces import compute_network_indices
from VertexTissue.visco_funcs import crumple

print('this is some basic output')



import numpy as np


import VertexTissue.globals as const
import VertexTissue.SG as SG


from VertexTissue.globals import belt_strength, inner_arc, outer_arc

from VertexTissue.Sweep import sweep
from VertexTissue.Tissue import  T1_minimal, get_outer_belt, tissue_3d
from VertexTissue.util import first
from VertexTissue.vertex_3d import monolayer_integrator

from VertexTissue.Energy import get_cell_volumes, network_energy

try:
    import matplotlib.pyplot as plt

    from VertexTissue.PyQtViz import edge_view, edge_viewer
    viewable=True
    base_path = './data/'
except:
    viewable=False
    base_path = '/scratch/st-jjfeng-1/lmackay/data/'

from VertexTissue.Dict import first_dict_value, last_dict_value
from VertexTissue.Geometry import euclidean_distance, unit_vector


def is_subprocess():
    return main_pid != os.getpid()

def decrease_nice():
    pid = os.getpid()
    os.system("sudo renice -n -19 -p " + str(pid))

dt=0.1
taus = np.logspace(6,1,5)
lvl=3
inter_edges = ((305, 248), (94,163), (69,8), (2,8))

def run(force, visco=False,  phi0=1.0, level=0, arcs=3, cable=True):

    #
    # G, G_apical = tissue_3d( gen_centers=T1_minimal,  basal=True)
    G, G_apical = tissue_3d( hex=7,  basal=True)

    save_dict={};
    save_dict[0]=copy.deepcopy(G)

    a=G.node[inter_edges[level][0]]['pos'];
    b=G.node[inter_edges[level][1]]['pos'];

    ab=unit_vector(a,b)

    G.node[inter_edges[level][1]]['pos'] = (a+b)/2 + .01*ab
    G.node[inter_edges[level][0]]['pos'] = (a+b)/2 - .01*ab
    save_dict[0.5]=copy.deepcopy(G)
    save_dict[1]=copy.deepcopy(G)
    # view=edge_viewer(G,attr='myosin')
    # view(G)

    return save_dict



    #     print(f'failed to integrate tau={tau}')
    
    #     pass

def shortest_length(d, e=inter_edges[lvl]):
    return min([ euclidean_distance(G.nodes[e[0]]['pos'],G.nodes[e[1]]['pos']) for G in d.values() ])

def plot_results(forces,results):
    if not viewable:
        return
    
    lens = results/const.l_apical
    # times = np.reshape([e[1] for e in results.ravel()], results.shape)
    inter_len = const.l_intercalation/const.l_apical

    phi = forces*const.myo_beta/const.l_apical
   
    for i in range(int(lens.shape[1]/2)):
        c=colors[i]
        if i>=1:
            prefix = f'$\phi_0$={phi0s[i-1]}'
            i=i+1
            j=i+len(phi0s)
        else:
            prefix='elastic'
            j=i+1

            
        plt.plot(phi, lens[:,i], color=c, label=prefix+' (cable)')
        plt.plot(phi, lens[:,j], linestyle='--', color=c)

    plt.xlabel(r'$-\phi^{\rm c}$', fontsize=16)
    plt.ylim((0,1))
    plt.ylabel(r'$\chi_{min}(\phi^{\rm c}| \phi_0 )$', fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()

    plt.savefig(f'triple_belt_lvl_{lvl}_contraction_127.pdf',dpi=200)
    plt.show()

    thresh=0.03
    
    psi_elastic=phi[first(lens[:,1]<thresh)][0]
    # phi_elastic=phi[first(lens[:,1]<thresh)][0]
    psi_cable = np.array([phi[first(lens[:,i+2]<thresh)][0] for i, _ in enumerate(phi0s)])
    psi_no_cable = np.array([phi[first(lens[:,i+2+len(phi0s)]<thresh)][0] for i, _ in enumerate(phi0s)])


    alpha_VE = (psi_elastic-psi_no_cable)/psi_elastic
    alpha_cable = (psi_no_cable-psi_cable)/psi_elastic

    plt.figure().clear()


    plt.plot(phi0s, alpha_VE, label=r'$\alpha_{\rm VE}$')
    plt.plot(phi0s, alpha_cable, label=r'$\alpha_{\rm cable}$')
    plt.plot(phi0s,alpha_cable+alpha_VE, label=r'$\alpha_{\rm tot}$')
    
    plt.xlabel('$\phi_0$', fontsize=16)
    plt.ylabel(r'$\alpha$', fontsize=16)

    np.save(f'viscoelastic_cable_advantange_triple_127_level_{lvl}.npy', alpha_cable )
    np.save(f'viscoelastic_advantange_triple_127_level_{lvl}.npy', alpha_VE )

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'viscoelastic_advantange_triple_127_level_{lvl}.pdf',dpi=200)
    plt.show()
    
    
    
def visco_runner(phi0, **kw):
    return lambda f: run(f, visco=True, phi0=phi0, **kw)

def visco_no_cable_runner(phi0, **kw):
    return lambda f: run(f, visco=True, cable=False, phi0=phi0, **kw)
# G_test, _ = tissue_3d( hex=7,  basal=True)
# belt = get_outer_belt(_)
# edges = get_myosin_free_cell_edges(G_test)
# nodes = np.unique(np.array([e for e in edges]).ravel())
# smth = [not inside_arc(n, inner_arc, G_test) for n in nodes]
# # inner_most = [inside_arc(n,outer_arc,G_test) for n in nodes]
# # smth = [inside_arc(n,outer_arc,G_test)  for n in nodes]
# bad_nodes = nodes[smth].ravel()

# def foo(d):
#     _, bar , min_len= shortest_edge_network_and_time(d,excluded_nodes=bad_nodes, return_length=True)
#     return min_len, bar



def energy_timeseries(d,phi0=1.0,**kw):
    G0=first_dict_value(d)
    _, _, _, _, triangle_inds, triangles_sorted, _ = compute_network_indices(G0) 
    triangulation= (triangle_inds, triangles_sorted)

    
    U_vec = np.array([ network_energy(G,phi0=phi0, ec=0.2, triangulation=triangulation, get_volumes=get_cell_volumes) for G in d.values()])

    # plt.plot(U_vec, marker ='.')
    # # plt.plot(U_vec_bis)
    # plt.show()

    return U_vec[-2]


phi0s=list(reversed([ .22, .25, .3,  .4, .5,  .6, .7, .8, .9]))
colors=['k','r','g','b','y','m','c','orange']

def main():
    forces=np.array([ *np.linspace(0,850,40), *np.linspace(850,1200,20)[1:]])
    forces = np.linspace(0, 800, 60)
    forces=[1200]
    # forces=np.linspace(0,850,60)




    
    kws={'cable':[False,], 'level':3, 'visco':[True,], 'phi0':phi0s}

    # import pickle 
    # with open('./data/Step0_bis/run/visco_phi0=0.7_level=3_600.pickle','rb') as file:
    #     d=pickle.load(file)
    # energy_timeseries(d)
    dw=sweep(forces, run,  kw = kws, pre_process=energy_timeseries, pass_kw=True, savepath_prefix=base_path, overwrite=False, cache=True, refresh=True)

    plt.plot(phi0s,dw[0,0,0,:])
    plt.show()

    # run(forces[-2], visco=True, phi0=0.95)
    print('done')

if __name__ == '__main__':
    main_pid = os.getpid()
    main()

