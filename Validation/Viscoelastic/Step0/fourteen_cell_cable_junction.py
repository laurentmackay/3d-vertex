import os

from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.lines as lines
print('this is some basic output')

from doctest import FAIL_FAST
from itertools import product
from pathos.pools import ProcessPool as Pool
import numpy as np


from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import T1_minimal, T1_shell, tissue_3d, get_outer_belt
import VertexTissue.SG as SG
import VertexTissue.T1 as T1

from VertexTissue.Analysis import parameter_sweep, parameter_sweep_analyzer

from VertexTissue.globals import default_ab_linker, default_edge, belt_strength
import VertexTissue.globals as const

from VertexTissue.util import first

try:
    from VertexTissue.PyQtViz import edge_view
    import matplotlib.pyplot as plt
    viewable=True
    base_path = './data/SAC+/'
except:
    viewable=False
    base_path = '/scratch/st-jjfeng-1/lmackay/data/SAC+/'

from VertexTissue.util import last_item
from VertexTissue.funcs import euclidean_distance, unit_vector


def is_subprocess():
    return main_pid != os.getpid()

def decrease_nice():
    pid = os.getpid()
    os.system("sudo renice -n -19 -p " + str(pid))

dt=0.1
taus = np.logspace(6,1,5)
def run(force, visco=False, cable=True, phi0=1.0):

    #
    G, G_apical = tissue_3d( gen_centers=T1_shell,  basal=True)
    
    belt = get_outer_belt(G_apical)

    # if visco:
    #     const.mu_apical*=phi0 


    #initialize some things for the callback
    # intercalation = T1.simple_T1(G, strength=force)
    # squeeze = SG.just_belt(G, belt, t=0, strength=force)
    squeeze = SG.belt_and_intercalation(G, belt, t_belt=0, inter_edges=((41,8),),  t_intercalate=0, intercalation_strength=force, belt_strength=belt_strength if cable else 0.0)
    def l_rest(ell,e,ec=.2):

            L0=G[e[0]][e[1]]['l_rest']

            if ell<(1-ec)*L0:
                    return (1-phi0)*ell/(1-ec)+ phi0*L0
            else:
                    return L0

    # const.press_alpha/=4
    if not visco:
        kw={}
    else:
        kw={'rest_length_func':l_rest}

    done=False
    def terminate(*args):
        nonlocal done
        done=True

    def wait_for_intercalation(*args):
        nonlocal done
        return done

    const.l_intercalation=0.1
    const.l_mvmt=0.001
    #create integrator
    integrate = monolayer_integrator(G, G_apical, 
                                    pre_callback=squeeze, 
                                    intercalation_callback=terminate, 
                                    termination_callback=wait_for_intercalation,  
                                    length_rel_tol=0.01, angle_tol=0.02, blacklist=True,
                                    player=False, viewer={'button_callback':terminate} if viewable else False, minimal=False, **kw)
    #integrates
    # try:
    # print(belt)
    if visco:
        pattern = f'visco_phi0={phi0}_{force}.pickle' if cable else f'visco_no_cable_phi0={phi0}_{force}.pickle' 
    else:
        pattern = f'elastic_edges_{force}.pickle' if cable else f'elastic_no_cable_{force}.pickle'

    pattern=base_path+pattern

    # pattern=None
    print(f'starting f={force}')
    dt_max=2 if force<300 else 0.5
    integrate(10, 3000, 
            dt_init = 1e-3,
            adaptive=True,
            dt_min=1e-2,
            save_rate=50,
            save_pattern=pattern)
    # except:
    #     print(f'failed to integrate tau={tau}')
    
    #     pass



def final_length(d):
    G = last_item(d)
    # edge_view(G)
    a = G.nodes[8]['pos']
    b = G.nodes[41]['pos']
    return euclidean_distance(a,b)

def shortest_length(d):
    lens = []
    for G in d.values():
        a = G.nodes[8]['pos']
        b = G.nodes[41]['pos']
        lens.append(euclidean_distance(a,b))
    
    # edge_view(G)

    return min(lens)

def collapsed_or_buckled(d):
    lens = []
    G=last_item(d)
    

    a = G.nodes[11]['pos']
    b = G.nodes[10]['pos']
    c = G.nodes[1]['pos']
    ab=unit_vector(a,b)
    ac=unit_vector(c,a)

    a = G.nodes[8]['pos']
    b = G.nodes[41]['pos']


    out=0
    if np.sum(ab*ac)<0:
        out+=0.5
    
    if euclidean_distance(a,b)<=const.l_intercalation:
        out+=1


    a = G.nodes[16]['pos']
    b = G.nodes[21]['pos']
    c = G.nodes[3]['pos']
    d = G.nodes[31]['pos']

    out+=1j*euclidean_distance(b,d)#*np.arccos(np.sum(unit_vector(a,b)*unit_vector(c,d)))

    return out

def buckled(d):
    lens = []
    G=last_item(d)
    
    a = G.nodes[11]['pos']
    b = G.nodes[10]['pos']
    c = G.nodes[7]['pos']
    ab=unit_vector(a,b)
    ac=unit_vector(a,c)
   
    
    # edge_view(G)

    return np.sum(ab*ac)>0

def plot_buckled(forces,results):

    fig=plt.figure()
    fig.set_size_inches(6, 5)
    phi = forces*const.myo_beta/const.l_apical
    # plt.imshow(results)
    data = np.array(results[:-1,2:-(len(phi0s)+1)].T, dtype=complex)
    buckle_data = np.real(data)
    splay_data = np.imag(np.array(results[:,2:-(len(phi0s))].T, dtype=complex))
    values = np.unique(buckle_data.ravel())
    values=values[~np.isnan(values)]
    contour_color='c'
    plt.contour(phi,phi0s,splay_data,levels=[2.2], colors=contour_color,origin='upper')
    im=plt.pcolormesh(phi, phi0s, buckle_data , shading='flat')
    
    colors = [ im.cmap(im.norm(value)) for value in values]
    lbls=['uncontracted','buckled+uncontracted','contracted','buckled+contracted']
    patches = [ mpatches.Patch(color=colors[i], label=lbls[i] ) for i, _ in enumerate(values) ]
    c_line = lines.Line2D([0], [0], color=contour_color, lw=2, label='cable collapse')
    plt.legend(handles=[*patches, c_line],  loc='center right')
    plt.xlabel(r'$-\phi^{\rm j}$', fontsize=16)
    plt.ylabel('$\phi_0$', fontsize=16)
    plt.tight_layout()



    plt.savefig('belt_junction_buckling.pdf',dpi=200)
    
    plt.show()

def plot_results(forces,results):
    if not viewable:
        return
    
    lens = results/const.l_apical
    inter_len = const.l_intercalation/const.l_apical

    
    # plt.plot(forces, lens[:,1],label='Elastic')
    # plt.plot(forces, lens[:,0],label='Elastic + Belt')
    # plt.plot(forces, lens[:,5],label='$\phi_0=0.8$ + Belt')
    # plt.plot(forces, lens[:,4],label='$\phi_0=0.6$ + Belt')
    # plt.plot(forces, lens[:,3],label='$\phi_0=0.45$ + Belt')
    # plt.plot(forces, lens[:,2],label='$\phi_0=0.3$ + Belt')
    phi = forces*const.myo_beta/const.l_apical
   
    for i, fi, c in zip(range(len(phi0_disp)),phi0_disp,colors):
        # if fi in phi0_disp:
        i=phi0s.index(fi)
        plt.plot(phi, lens[:,2+i],linestyle='-',color=c, label=f'$\phi_0={fi}$')
        # plt.plot(phi, lens[:,2+i+len(phi0s)],linestyle='--',color=c)

    plt.plot(phi, lens[:,0],linestyle='-',color='k', label=f'elastic')
    plt.xlim((0,1.4))
    plt.ylim((0,1))
    plt.plot([phi[0], phi[-1]],[inter_len, inter_len],':k')

    plt.xlabel(r'$-\phi^{\rm j}$', fontsize=16)
    plt.ylabel(r'$\chi(\phi^{\rm j}, -2.2| \phi_0 )$', fontsize=16)
    # plt.plot(forces, lens[:,2],'-r',label='$\phi_0=0.3$')
    # plt.plot(forces, lens[:,6],'--r')
    # plt.plot(forces, lens[:,3],'-g',label='$\phi_0=0.45$')
    # plt.plot(forces, lens[:,7],'--g')
    # plt.plot(forces, lens[:,4],'-b',label='$\phi_0=0.6$')
    # plt.plot(forces, lens[:,8],'--b')
    # plt.plot(forces, lens[:,5],'-y',label='$\phi_0=0.8$')
    # plt.plot(forces, lens[:,9],'--y')

    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('belt_junction_contraction.pdf',dpi=200)
    plt.show()
    
    plt.figure().clear()

    thresh=0.032
    adv=[]
    for i, fi in enumerate(phi0s):
        i = 2+i
        j = i+len(phi0s)
        delta = phi[first(lens[:,j]<thresh)]-phi[first(lens[:,i]<thresh)]
        adv.append(delta)

    plt.plot(phi0s,adv)
    plt.xlabel('$\phi_0$', fontsize=16)
    plt.ylabel(r'$\phi_{0,{\rm eff}}^{{\rm cable}}-\phi_{0,{\rm eff}}^{{\rm no cable}}$', fontsize=16)

    plt.tight_layout()
    plt.savefig('belt_junction_advantange.pdf',dpi=200)
    plt.show()



    psi_elastic=phi[first(lens[:,1]<thresh)][0]
    psi_cable = np.array([phi[first(lens[:,i+2]<thresh)][0] for i, _ in enumerate(phi0s)])
    psi_no_cable = np.array([phi[first(lens[:,i+2+len(phi0s)]<thresh)][0] for i, _ in enumerate(phi0s)])

    alpha_VE = (psi_elastic-psi_no_cable)/psi_elastic
    alpha_cable = (psi_no_cable-psi_cable)/psi_elastic

    plt.figure().clear()
    plt.plot(phi0s,alpha_VE, label=r'$\alpha_{\rm VE}$')
    plt.plot(phi0s,alpha_cable, label=r'$\alpha_{\rm cable}$')
    plt.plot(phi0s,alpha_cable+alpha_VE, label=r'$\alpha_{\rm tot}$')

    plt.xlabel('$\phi_0$', fontsize=16)
    plt.ylabel(r'$\alpha$', fontsize=16)
    plt.ylim((0,1))
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig('visco_cable_advantange.pdf',dpi=200)
    plt.show()

def visco_runner(phi0):
    return lambda f: run(f, visco=True, phi0=phi0)

def visco_no_cable_runner(phi0):
    return lambda f: run(f, visco=True, cable=False, phi0=phi0)
phi0_disp=[0.3, .42, .5,  0.6, .7, .8,  .9]
phi0s=[0.3, .34, .38, .42, .45, .5, .51, .52, .53, .54, .55, .56, .57, .58, .59, 0.6, .63, .67, .7, .73, .77, .8, .85, .9, .95]
colors=['r','orange','y','g','c','b','m']
def main():
    forces=np.linspace(0,600,80)
    
    # visco_conds=product((True,), forces, phi0)
    # elastic_conds=product((False,), forces)
    
    visco_funcs=[ *[visco_runner(phi) for phi in phi0s],
                    *[visco_no_cable_runner(phi) for phi in phi0s]]



    elastic_func=run
    funcs=[elastic_func, lambda f: run(f, cable=False), *visco_funcs]
    savepaths = [
        [base_path+f'elastic_edges_{force}.pickle',
        base_path+f'elastic_no_cable_{force}.pickle',
        *[base_path+f'visco_phi0={phi0}_{force}.pickle' for phi0 in phi0s],
        *[base_path+f'visco_no_cable_phi0={phi0}_{force}.pickle' for phi0 in phi0s]
         ] for force in forces
    ]

    if viewable:
        results =parameter_sweep(forces, funcs, pre_process=shortest_length, savepaths=savepaths, overwrite=False, inpaint=np.nan, cache=True)
        plot_results(forces, results)
    else:
        parameter_sweep(forces, funcs, savepaths=savepaths, overwrite=False)
    # run(580, visco=False, phi0=0.6)
    print('done')

if __name__ == '__main__':
    main_pid = os.getpid()
    main()

