

from util import mkprocess

import networkx as nx
import numpy as np

import multiprocessing as mp

from pyqtgraph.Qt import  QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from GLNetworkItem import GLNetworkItem
from globals import basal_offset

def edge_viewer(*args, refresh_rate=60,**kwargs):

    def outer(*a,**kw):    
        def inner(b):
            gi = edge_view(*a,**kw)
            def listen():
                if b.poll():
                    while b.poll():
                        G=b.recv()
                    edge_view(G, gi=gi,**kw)


            tmr = QtCore.QTimer()
            tmr.timeout.connect(listen)
            
            tmr.setInterval(500/refresh_rate)
            tmr.start()

            pg.exec()

        return inner


    a,b = mp.Pipe(duplex=True)
    proc = mkprocess(outer(*args,**kwargs), args=(b,))
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


def _init_window(title=''):

        return w


def edge_view(G, gi=None, cell_edges_only=False, exec=False, attr=None, colormap='CET-D4',edgeWidth=1.25, edgeWidthMultiplier=3, edgeColor=0.0):
    pos = np.array([*nx.get_node_attributes(G,'pos').values()])

    if gi is None:
        pg.mkQApp("Edge View")
        w = gl.GLViewWidget()
        w.setBackgroundColor(1.0)
        w.show()
        w.setCameraPosition(distance=3*np.sqrt(np.sum((pos*pos),axis=1)).max())

    ind_dict={n:i for i,n in enumerate(G._node)}

    if cell_edges:
        circum_sorted = G.graph['circum_sorted']
        apical = np.vstack([np.array([[c[i-1],c[i]] if c[i-1]<c[i] else [c[i],c[i-1]]  for i, _ in enumerate(c)]) for c in  circum_sorted])
        basal = apical+basal_offset
        ab = np.vstack([[n, n+basal_offset] for n in np.unique(apical)])
        edges = np.vstack((apical,basal,ab))
    else:
        edges=G.edges()

    

    cmap = pg.colormap.get(colormap)
    
    edgeColor=pg.glColor(edgeColor)

    if attr:
        attrs=nx.get_edge_attributes(G,attr)
        if cell_edges:
            vals=np.array([attrs[(e[0],e[1])] for e in edges])
        else:
            vals = np.fromiter(attrs.values(),dtype=float)
        vals = (vals-vals.min())
        range   = vals.max()
        if range:
            vals = vals/range
            edgeColor = cmap.mapToFloat(vals)
            edgeWidth = edgeWidth*(1+vals*edgeWidthMultiplier)
        else:
            edgeColor = cmap.mapToFloat(0.0)


    edges=np.array([np.array([ ind_dict[e[0]], ind_dict[e[1]]]) for e in edges])
    data = {'edges':edges, 'edgeColor':edgeColor, 'nodePositions':pos, 'edgeWidth':edgeWidth, 'nodeSize':0.0}

    if gi is None:
        gi = GLNetworkItem(**data)
        w.addItem(gi)
    else:
        gi.setData(**data)

    if exec:
        pg.exec()

    return gi
    


