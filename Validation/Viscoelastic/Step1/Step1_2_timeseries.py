import numpy as np
import matplotlib.pyplot as plt

from VertexTissue.Sweep import sweep


from Step1 import extending_edge_length, run, phi0s,  base_path, middle_edge, outer_edge
import Step1_1
from Step1_1 import  bar, kws_middle, kws_middle_inner_only, kws_outer_no_cable
from VertexTissue.Player import pickle_player





def extension_timeseries(d, edge=None):
    return np.array([(t, extending_edge_length(d[t], edge=edge)) for t in d])
    

def foo(d, edge=None):
    # t=np.array([*d.keys()])
    ell = extension_timeseries(d, edge=edge)
    # plt.plot(t,ell)
    # plt.show()
    return ell[-1]

fontsize=14

if __name__ == '__main__':

        
    

    lens=sweep(phi0s, run, kw=kws_outer_no_cable, pre_process=extension_timeseries, pre_process_kw={'edge': outer_edge}, 
                  cache=True,  savepath_prefix=base_path, inpaint=np.nan)

    for entry, phi0 in zip(lens, phi0s):
        plt.plot(entry[0][:,0], entry[0][:,1], label=f'$\phi_0={phi0}$')

    plt.xlim(0,15000)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('New Edge Length ($\mu$m)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'extension_timeseries.pdf',dpi=200)
    plt.show()




