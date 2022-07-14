import os



import VertexTissue.SG as SG
import VertexTissue.globals as const

from VertexTissue.Tissue import get_outer_belt, tissue_3d
from VertexTissue.globals import belt_strength, outer_arc, inner_arc
from VertexTissue.util import script_name, signature_string
from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.visco_funcs import crumple, extension_remodeller, shrink_edges






try:
    from VertexTissue.PyQtViz import edge_view
    import matplotlib.pyplot as plt
    viewable=True
    base_path = './data/Step1.1/'
except:
    viewable=False
    base_path = '/scratch/st-jjfeng-1/lmackay/data/Step1.1/'


def run(force, visco=False, cable=True, phi0=1.0, level=0, arcs=2):

    #
    G, G_apical = tissue_3d( hex=7,  basal=True)
    
    belt = get_outer_belt(G_apical)

    # if visco:
    #     const.mu_apical*=phi0 


    #initialize some things for the callback
    # intercalation = T1.simple_T1(G, strength=force)
    # squeeze = SG.just_belt(G, belt, t=0, strength=force)
    
    if arcs==1:
        arc_list=(outer_arc, )
    elif arcs==2:
        arc_list=(outer_arc, inner_arc)
    #
    squeeze = SG.arc_pit_and_intercalation(G, belt,
                                         arc_strength=belt_strength if cable else 0.0,
                                         t_intercalate=0, t_1=0, intercalation_strength=force, pit_strength=0)

    if not visco:
        kw={}
    else:
        kw={'rest_length_func': crumple(phi0=phi0)}

    done=False
    def terminate(*args):
        nonlocal done
        done=True

    def wait_for_intercalation(*args):
        nonlocal done
        return done

    
    #create integrator
    integrate = monolayer_integrator(G, G_apical, 
                                    pre_callback=squeeze, 
                                    blacklist=True, RK=1,
                                    intercalation_callback=shrink_edges(G, L0_min=0.5),
                                     angle_tol=2, length_rel_tol=0.01,
                                     maxwell=True, maxwell_nonlin=extension_remodeller(),
                                    player=False, viewer={'button_callback':terminate, 'nodeLabels': None} if viewable else False, minimal=False, **kw)


    pattern=os.path.join(base_path, script_name(), signature_string()+'.pickle')

    # pattern=None
    print(f'starting f={force}')

    integrate(0.5, 3000, 
            dt_init = 1e-3,
            adaptive=True,
            dt_min=1e-2,
            save_rate=50,
            verbose=True,
            save_pattern=pattern)



def main():
    run(750, visco=False, phi0=0.3, cable=True)

if __name__ == '__main__':
    main()


