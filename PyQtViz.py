import multiprocessing as mp
import inspect
import time


import networkx as nx
import numpy as np

from pyqtgraph.Qt.QtGui import *
from pyqtgraph.Qt.QtCore import *
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget


from util import mkprocess, get_creationtime, get_filenames
from GLNetworkItem import GLNetworkItem
from globals import basal_offset, save_pattern


def edge_viewer(*args, refresh_rate=60, parallel=True, **kw):

    def outer(*a,**kw):
        init_callback = kw['window_callback'] if 'window_callback' in kw.keys() and callable(kw['window_callback']) else lambda win: None
        win=None
        gi=None
        G=None
        wind=None
        glview=None

        def modify_bool_kw(key):
            def inner_bool(s):
                nonlocal kw
                kw={**kw, **{key:s!=0}}
                draw()

            return inner_bool

        def modify_float_kw(key):
            def inner_float(v):
                nonlocal kw
                if len(v)==0:
                    return
                if v[0]=='.':
                    v='0'+v
                if v[-1]=='.':
                    v=v+'0'
                try:
                    kw={**kw, **{key:float(v)}}
                    draw()
                except:
                    pass
            return inner_float


        def mk_controls(win):
            nonlocal wind, glview
            wind=win
            _, glview = win.children()

            docker = QDockWidget(win)
            widget = QWidget()
            
            checkbox_layout = QHBoxLayout()

            layout = QVBoxLayout()


            argspec=inspect.getargspec(edge_view)
            args=argspec.args
            defs=argspec.defaults
            ind_delta = len(args)-len(defs)
            keyword_value = lambda k : defs[args.index(k)-ind_delta] if k not in kw.keys() else kw[k]


            bools={'apical_only':'Apical Only', 'cell_edges_only':'No Spokes'}
            for k, v in bools.items():
                box=QCheckBox(v)
                box.setCheckState(2 if keyword_value(k) else 0)
                box.stateChanged.connect(modify_bool_kw(k))
                checkbox_layout.addWidget(box)

            form_layout = QFormLayout()
            
            floats = {'edgeWidth':'Edge Width', 'edgeWidthMultiplier': 'Edge Width Mult.', 'spokeAlpha':'Spoke Alpha'}
            for k, v in floats.items():
                line=QLineEdit()
                valid = QDoubleValidator(0, float('inf'), 3)
                valid.setNotation(QDoubleValidator.StandardNotation)
                line.setValidator(valid)
                line.setText(str(keyword_value(k)))
                line.textChanged.connect(modify_float_kw(k))
                form_layout.addRow(QLabel(v), line)

            layout.addLayout(checkbox_layout)
            layout.addLayout(form_layout)

            widget.setLayout(layout)

            docker.setWidget(widget)
            docker.setFloating(False)

            win.addDockWidget(Qt.RightDockWidgetArea, docker)
            init_callback(win)

        
        kw = {**kw, **{'window_callback':mk_controls}}

        if not parallel:
            G=a[0]
            gi = edge_view(*a, **kw)

        def draw(*a, **kw2):
            nonlocal kw, G, wind, glview
            if len(a):
                G=a[0]

            if gi and G:
                edge_view(G, gi=gi, **{**kw, **kw2})

        def start_in_parallel(b):
            nonlocal G, gi

            G=a[0]
            gi = edge_view(*a, **kw)

            def listen():
                nonlocal G

                if b.poll():
                    while b.poll(): #empty the pipe if there has been some accumulatio
                        (G,kw2)=b.recv()
                        received = True
                    draw(G,**kw2)
                    b.send([])

            b.send([])
            tmr = QTimer()
            tmr.timeout.connect(listen)
            
            tmr.setInterval(500/refresh_rate)
            tmr.start()

            pg.exec()

        if parallel:
            return start_in_parallel
        else:
            return draw
            

    plot=True
    if parallel:
        a, b = mp.Pipe(duplex=True)
        proc = mkprocess(outer(*args,**kw), args=(b,))
        proc.start()

        
        def pipe_plot(G, callback=None,**kw):
            nonlocal plot
            if plot:
                try:
                    if a.poll(0.5/refresh_rate): #wait up to half a framecycle to see if the plotter thread is ready to receive
                        _ = a.recv() 
                        a.send((G,kw))
                except:
                    print('plotting is no longer an option')
                    plot=False


        return pipe_plot

    else:
        return outer(*args,**kw)

    

def edge_view(G, gi=None, size=(640,480), cell_edges_only=True, apical_only=False, exec=False, attr=None, colormap='CET-D4', edgeWidth=1.25, edgeWidthMultiplier=3, spokeAlpha=.15, edgeColor=0.0, title="Edge View", window_callback=None):
    

    pos = np.array([*nx.get_node_attributes(G,'pos').values()])

    if gi is None:
        # if  QApplication.instance() is None:
        app = pg.mkQApp(title)

        win = QMainWindow()
        
        w = GLViewWidget(parent=win)
        w.setMinimumSize(QSize(*size))
        w.setBackgroundColor(1.0)
        w.show()
        w.setCameraPosition(distance=3*np.sqrt(np.sum((pos*pos),axis=1)).max())
        win.setCentralWidget(w)
        win.show()
        if callable(window_callback):
            window_callback(win)

    ind_dict={n:i for i,n in enumerate(G._node)}

    # if cell_edges_only:
    circum_sorted = G.graph['circum_sorted']
    apical = np.vstack([np.array([[c[i-1],c[i]] if c[i-1]<c[i] else [c[i],c[i-1]]  for i, _ in enumerate(c)]) for c in  circum_sorted])
    
    if apical_only:
        edges = (apical,)
    else:
        basal = apical+basal_offset
        ab = np.vstack([[n, n+basal_offset] for n in np.unique(apical)])
        edges = (apical,basal,ab)

    if not cell_edges_only:
        centers = G.graph['centers']
        spokes = np.vstack([np.array([[o,ci] if ci>o else [ci,o]  for ci in c]) for o, c in  zip(centers,circum_sorted)])
        if not apical_only:
            spokes=(spokes,spokes+basal_offset)
        else:
            spokes=(spokes,)

        edges=edges+spokes

    edges=np.vstack(edges)

    cmap = pg.colormap.get(colormap)
    
    edgeColor=pg.glColor(edgeColor)

    if attr:
        attrs=nx.get_edge_attributes(G,attr)

        vals=np.array([attrs[(e[0],e[1])] for e in edges])
        vals = (vals-vals.min())
        range   = vals.max()

    elif not cell_edges_only:
        vals = np.zeros((edges.shape[0],1))
        range=false


    if range:
        vals = vals/range

    edgeColor = cmap.mapToFloat(vals)
    edgeWidth = edgeWidth*(1+vals*edgeWidthMultiplier)
    if not cell_edges_only:
        edgeColor[-sum(tuple(map(len, spokes))):,3]=spokeAlpha



    edges=np.array([np.array([ ind_dict[e[0]], ind_dict[e[1]]]) for e in edges])

    data =  {'edges':edges, 'edgeColor':edgeColor, 'nodePositions':pos, 'edgeWidth':edgeWidth, 'nodeSize':0.0}

    if gi is None:
        gi = GLNetworkItem(parent=w, **data)
        w.addItem(gi)
        gi.setParent(w)
    else:
        gi.setData(**data)

    gi.parent().parent().setWindowTitle(title)
    
    if exec:
        pg.exec()

    return gi



