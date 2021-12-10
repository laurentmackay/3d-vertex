import networkx as nx
import numpy as np
import multiprocessing as mp
from mayavi import mlab
from util import run_dill, make_dill
import time
import vtk

vtk.vtkObject.GlobalWarningDisplayOff()

def edge_viewer(*args,**kwargs):
    a,b = mp.Pipe(duplex=True)
    man = _view_edges(*args,**kwargs)
    proc = mp.Process(target=run_dill, args=make_dill(man, b), daemon=True)
    proc.start()
    return a.send

def _view_edges(*a,**kw):    
    def inner(b):
        pts = edge_view(*a,**kw)
        @mlab.animate(delay=16,ui=False)
        def anim():
            while True:
                if b.poll():
                    while b.poll():
                        msg=b.recv()
                    edge_view(msg, *a[1:-1], pts=pts, **kw)
                
                yield
        an=anim()
        mlab.show()

    return inner



def edge_view(G, pts=None, tube_radius=0.15, colormap='Blues', attr=None, color=None, radius_factor = 3):
    draw = pts is None
    if draw:
        mlab.figure(figure=1, bgcolor=(0.3,0.3,0.3), size=(800, 600))
    else:
        pts.scene.disable_render = True

    pos = nx.get_node_attributes(G,'pos')

    x = []
    y = []
    z = []
    connections = []
    if attr is not None:
        s = []
    

    for i, e in enumerate(G.edges()):
        

        edge_xyz = np.array([pos[e[0]],pos[e[1]]])

        x.append(edge_xyz[:, 0])
        y.append(edge_xyz[:, 1])
        z.append(edge_xyz[:, 2])
        connections.append(np.array([2*i,2*i+1]))
        if attr is not None:
            scalar = G.get_edge_data(*e)[attr]
            s.extend((scalar,scalar))

        

    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)
    s = np.hstack(s)
    
    if draw:
        if attr is not None:
            args = (x,y,z,s)
        else:
            args= (x,y,z)
        pts = mlab.pipeline.scalar_scatter(*args,
                            scale_factor=0.1,
                            scale_mode='none',
                            colormap=colormap,
                            resolution=20)
    else:
        pts.mlab_source.set(x=x,y=y,z=z,scalars=s)

    pts.mlab_source.dataset.lines = np.vstack(connections)
    
   
    if draw:
        
        tube = mlab.pipeline.tube(pts, tube_radius=tube_radius)
        mlab.pipeline.surface(tube, color=color)
        tube.filter.radius_factor = radius_factor
        tube.filter.vary_radius = 'vary_radius_by_scalar'
        # pts.remove()


    pts.scene.disable_render = False
    if not draw:
        pts.scene.render()
        # 
    
   

        

    return pts
    
    

  #start visualisation loop in the main-thread, blocking other executions

        # loopThread.join()



