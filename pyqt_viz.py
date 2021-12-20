import sys
IS_WINDOWS = sys.platform.startswith('win')

if IS_WINDOWS:
    from util import run_dill, make_dill

import networkx as nx
import numpy as np
import multiprocessing as mp

from pyqtgraph.Qt import  QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL.GL import *
from OpenGL.GLUT import *
from GLNetworkItem import GLNetworkItem

def edge_viewer(*args, refresh_rate=60,**kwargs):
    a,b = mp.Pipe(duplex=True)
    if IS_WINDOWS:
        proc = mp.Process(target=run_dill, args=make_dill(_view_edges(*args, refresh_rate=refresh_rate,**kwargs), b), daemon=True)
    else:
        proc = mp.Process(target=_view_edges(*args,refresh_rate=refresh_rate,**kwargs), args=(b,), daemon=True)
    proc.start()

    plot=True
    def safe_plot(G):
        nonlocal plot
        if plot:
            try:
                a.send(G)
            except:
                plot=False

    return safe_plot

def _view_edges(*a,refresh_rate=60,**kw):    
    def inner(b):
        gi = edge_view(*a,**kw)
        def listen():
            if b.poll():
                while b.poll():
                    G=b.recv()
                edge_view(G, gi=gi,**kw)
            # print('looper')

        tmr = QtCore.QTimer()
        tmr.timeout.connect(listen)
        
        tmr.setInterval(500/refresh_rate)
        tmr.start()

        
        pg.exec()

    return inner



def edge_view(G, gi=None, exec=False, attr=None, colormap='CET-D4',edgeWidth=1.25, edgeWidthMultiplier=3):
    pos = np.array([*nx.get_node_attributes(G,'pos').values()])
    if gi is None:
        app = pg.mkQApp("Edge View")
        
        w = gl.GLViewWidget()
        w.setBackgroundColor(1.0)
        w.show()
        w.setCameraPosition(distance=3*np.sqrt(np.sum((pos*pos),axis=1)).max())
    

    
    


    ind_dict={n:i for i,n in enumerate(G._node)}
    edges=np.array([np.array([ ind_dict[e[0]], ind_dict[e[1]]]) for e in G.edges()])
    
    cmap = pg.colormap.get(colormap)
    edgeColor=pg.glColor("w")

    if attr:
        vals = np.fromiter(nx.get_edge_attributes(G,attr).values(),dtype=float)
        vals = (vals-vals.min())
        range   = vals.max()
        if range:
            vals = vals/range
            edgeColor = cmap.mapToFloat(vals)
            edgeWidth = edgeWidth*(1+vals*edgeWidthMultiplier)
        else:
            edgeColor = cmap.mapToFloat(0.0)
    
    if gi is None:
        gi = GLNetworkItem(edges=edges,
        edgeColor=edgeColor,
        nodePositions=pos,
        edgeWidth=edgeWidth,
        nodeSize=0.0)
        w.addItem(gi)
    else:
        gi.setData(edges=edges,
        edgeColor=edgeColor,
        nodePositions=pos,
        edgeWidth=edgeWidth,
        nodeSize=0.0)
        # QtGui.QApplication.processEvents()

    return gi
    
    

  #start visualisation loop in the main-thread, blocking other executions

        # loopThread.join()



