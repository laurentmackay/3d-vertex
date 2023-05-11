import multiprocessing as mp
import inspect
import time


import networkx as nx
import numpy as np

from pyqtgraph.Qt.QtWidgets import QMainWindow, QDockWidget, QWidget, QHBoxLayout, QVBoxLayout, QCheckBox, QFormLayout, QLineEdit, QLabel, QPushButton
from pyqtgraph.Qt.QtGui import QDoubleValidator
from pyqtgraph.Qt.QtCore import QSize, Qt, QTimer
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget


from .Multiprocessing import Process
from .GLNetworkItem import GLNetworkItem



def edge_viewer(*args, refresh_rate=10, parallel=True, drop_frames=True, button_callback=None, title=None, **kw):

    def outer(*a,**kw):
        init_callback = kw['window_callback'] if 'window_callback' in kw.keys() and callable(kw['window_callback']) else lambda win: None
        win=None
        gi=None
        colorbar=None
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

        def modify_string_kw(key):
            def inner_string(v):
                nonlocal kw
                if len(v)==0:
                    return
                try:
                    kw={**kw, **{key:v}}
                    draw()
                except:
                    pass
            return inner_string



        def mk_controls(win):
            nonlocal wind, glview, colorbar
            wind=win
            _, glview = win.children()

            docker = QDockWidget(win)
            widget = QWidget()
            
            

            layout = QVBoxLayout()


            argspec=inspect.getfullargspec(edge_view)
            args=argspec.args
            defs=argspec.defaults
            ind_delta = len(args)-len(defs)
            keyword_value = lambda k : defs[args.index(k)-ind_delta] if k not in kw.keys() else kw[k]

            


            checkbox_layout = QHBoxLayout()

            bools={'apical':'Apical', 'basal':'Basal', 'cell_edges_only':'No Spokes'}
            for k, v in bools.items():
                box=QCheckBox(v)
                box.setCheckState(2 if keyword_value(k) else 0)
                box.stateChanged.connect(modify_bool_kw(k))
                checkbox_layout.addWidget(box)

            form_layout = QFormLayout()

            strings = {'attr':'Edge Attribute'}
            for k, v in strings.items():
                line=QLineEdit()
                line.setText(str(keyword_value(k)))
                line.textChanged.connect(modify_string_kw(k))
                form_layout.addRow(QLabel(v), line)
            
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

            if button_callback is not None:
                button = QPushButton()
                button.setCheckable(True)

       
        
        
                button.setEnabled(True)
                button.setText('run callback')
                def request_callback_execution():
                    b.send('run callback')

                button.clicked.connect(request_callback_execution)

                layout.addWidget(button)

            widget.setLayout(layout)

            docker.setWidget(widget)
            docker.setFloating(False)

            win.addDockWidget(Qt.RightDockWidgetArea, docker)
            init_callback(win)

        
        kw = {**kw, **{'window_callback':mk_controls}}

        if not parallel:
            G=a[0]
            gi = edge_view(*a, **kw)
            # colorbar=gi.gview
            # glview = 

        def draw(*a, **kw2):
            nonlocal kw, G, wind, glview
            if len(a):
                G=a[0]

            if gi and G:
                edge_view(G, gi=gi, **{**kw, **kw2})

        def start_in_parallel(b):
            nonlocal G, gi, colorbar

            G=a[0]
            gi = edge_view(*a, **kw)
            # colorbar=gi.colorBar
            # wtv = gi.gview
            def listen():
                nonlocal G

                if b.poll():
                    while b.poll(): #empty the pipe if there has been some accumulation
                        (G,kw2)=b.recv()
                        received = True
                        
                    draw(G,**kw2)
                    b.send([])

            b.send([])
            tmr = QTimer()
            tmr.timeout.connect(listen)
            
            tmr.setInterval(int(500/refresh_rate))
            tmr.start()

            pg.exec()

        if parallel:
            return start_in_parallel
        else:
            return draw
            

    plot=True
    if parallel:
        a, b = mp.Pipe(duplex=True)
        proc = Process(outer(*args,**kw), args=(b,), daemon=True)
        proc.start()

        timeout =  0.0 if drop_frames else 0.5/refresh_rate 
        def pipe_plot(G, **kw):
            nonlocal plot
            if plot:
                try:
                    if a.poll(timeout): #wait up to half a framecycle to see if the plotter thread is ready to receive
                        msg = a.recv() 
                        a.send((G,kw))

                        # print(msg)
                        if len(msg)>0  and  msg=='run callback':
                            button_callback()
                    else:
                        print(f'not ready {timeout}')
                except:
                    print('plotting is no longer an option')
                    plot=False

        def kill():
            proc.kill()

        pipe_plot.kill=kill
        return pipe_plot

    else:
        return outer(*args,**kw)

class myGraphicsView(pg.GraphicsView):
    def __init__(self, parent=None, useOpenGL=None, background='default', padding=(5,10)):
        self.pixelPadding=padding
        super().__init__(parent=parent, useOpenGL=useOpenGL, background=background)
        

    def resizeEvent(self, ev):
        super().resizeEvent(ev)

        pg.GraphicsView.setRange(self, self.range, padding=(self.pixelPadding[0]/self.visibleRange().width(),-self.pixelPadding[1]/self.visibleRange().height()), disableAutoPixel=False)  ## we do this because some subclasses like to redefine setRange in an incompatible way.
        self.updateMatrix()

def edge_view(G, gi=None, size=(640,480), cell_edges_only=True, apical=True, basal=False, exec=False, attr=None, label_nodes=True, colormap='CET-D8', vmin=None, vmax=None, edgeWidth=1.25, edgeWidthMultiplier=3, spokeAlpha=.15, edgeColor=0.0, title="Edge View", window_callback=None, **kw):

    has_basal = 'basal_offset' in G.graph.keys()
    if has_basal:
        basal_offset=G.graph['basal_offset']

    pos = np.array([*nx.get_node_attributes(G,'pos').values()])

    if gi is None:
        # if  QApplication.instance() is None:
        app = pg.mkQApp(title)

        win = QMainWindow()
        layout = pg.LayoutWidget()
        layout.layout.setContentsMargins(0,0,0,0)
        layout.layout.setHorizontalSpacing(0)
        gl_widget = GLViewWidget(parent=win)
        gl_widget.setMinimumSize(QSize(*size))
        gl_widget.setBackgroundColor(1.0)
        gl_widget.show()
        gl_widget.setCameraPosition(distance=3*np.sqrt(np.sum((pos*pos),axis=1)).max())
        win.setCentralWidget(layout)
        layout.addWidget(gl_widget)
        win.show()
        if callable(window_callback):
            window_callback(win)

    ind_dict_0={n:i for i,n in enumerate(G._node)}

    # if cell_edges_only:
    if 'circum_sorted' in G.graph.keys():
        circum_sorted = G.graph['circum_sorted']
        apical_edges = np.vstack([np.array([[c[i-1],c[i]] if c[i-1]<c[i] else [c[i],c[i-1]]  for i, _ in enumerate(c)]) for c in  circum_sorted])

        if apical and not basal:
            edges = (apical_edges,)

        if basal and has_basal:
            basal_edges = apical_edges+basal_offset
            if apical:
                ab_edges = np.vstack([[n, n+basal_offset] for n in np.unique(apical_edges)])
                edges = (apical_edges, basal_edges, ab_edges)
            else:
                edges=(basal_edges,)
        
        # if apical and 

        
        # if apical_only or not has_basal:
        #     edges = (apical,)
        # else:
        #     basal = apical+basal_offset
            
        #     edges = (apical,basal,ab)
        apical_only = apical and not basal

        if not cell_edges_only:
            centers = G.graph['centers']
            spokes = np.vstack([np.array([[o,ci] if ci>o else [ci,o]  for ci in c]) for o, c in  zip(centers,circum_sorted)])

            if apical and basal:
                spokes=(spokes, spokes+basal_offset)
            elif not apical:
                spokes=(spokes+basal_offset,)
            elif not basal:
                spokes=(spokes,)

            if apical or basal:
                edges=edges+spokes
    else:
        edges=G.edges()

    edges=np.vstack(edges)

    visble_nodes = np.unique(edges.flatten())
    ind_dict={n:i for i,n in enumerate(visble_nodes)}
    visble_nodes = np.array([ind_dict_0[n] for n in visble_nodes])
    visible_pos = pos[visble_nodes]

    
    if label_nodes:
        lbls = [str(n) for n in visble_nodes]
    else:
        lbls = None

    cmap = pg.colormap.get(colormap)
    
    # edgeColor=pg.glColor(edgeColor)

    if attr:
        if type(attr) is dict:
            attrs=nx.get_edge_attributes(G,list(attr.keys())[0])
            attr_fun = list(attr.values())[0]
            # attrs ={k: attr_fun(v) for k,v in attrs.items()}
        else:
            attrs=nx.get_edge_attributes(G,attr)
        
        vals=np.array([attrs.get((e[0],e[1])) if attrs.get((e[0],e[1])) is not None else np.NAN for e in edges])
       
        if type(attr) is dict:
            vals = attr_fun(vals)
        
        
        if vmin is not None:
            min_val=vmin
        else:
            min_val = np.nanmin(vals)

        vals = (vals-min_val)

        if vmin is not None:
            vals[vals<0]=0
        

        range   = np.nanmax(vals)

    else:
        vals = np.zeros((edges.shape[0],))
        min_val=0.0
        range=False



    if range:
        vals = vals/range
    # stops, colors = cmap.getStops()
    # max_val = min_val + range
    # mid=int(np.floor(len(stops)/2))
    # zero=-min_val/range
    # stops=np.array([*np.linspace(0,zero,mid), *(np.linspace(zero,1.0,len(stops)-mid+1)[1:])])

    # cmap=pg.colormap.ColorMap(stops, colors)

    edgeColor = cmap.mapToFloat(vals)
    edgeWidth = edgeWidth*(1+vals*edgeWidthMultiplier)
    if not cell_edges_only:
        edgeColor[-sum(tuple(map(len, spokes))):,3]=spokeAlpha



    edges=np.array([np.array([ ind_dict[e[0]], ind_dict[e[1]]]) for e in edges])

    data =  {'edges':edges, 'edgeColor':edgeColor, 'nodePositions':visible_pos, 'edgeWidth':edgeWidth, 'nodeSize':0.0, 'nodeLabels':lbls}

    if gi is None:
        gi = GLNetworkItem(parent=gl_widget, **{ **data, **kw})
        gl_widget.addItem(gi)       
        gi.setParent(gl_widget)
        # if attr:
        colorBar = pg.ColorBarItem(values=(min_val, min_val+range), width=15, cmap=cmap, interactive=False)
        # colorBar.setGradient(cmap.getGradient())
        # labels = {"wtv": v for v in np.linspace(0, 1, 4)}
        # colorBar.setLabels(labels)
        gview = myGraphicsView(background='w')
        gview.setMinimumSize(QSize(80,size[1]))
        gview.setMaximumSize(QSize(80,10000))
        layout.addWidget(gview)
        gview.setCentralItem(colorBar)
        # gview.get
        # colorBar.setParentItem(gview)
        # win.addItem(colorBar)
        gi.colorBar=colorBar
        gi.gview = gview
    else:
        gi.setData(**{ **data, **kw})
        if not np.isnan(min_val) and not np.isnan(range):
            if range==0.:
                range=1.0
            gi.colorBar.setLevels(values=(min_val, min_val+range))
            # gi.colorBar.setColorMap(cmap)

    gi.parent().parent().parent().setWindowTitle(title)
    
    if exec:
        pg.exec()

    return gi



