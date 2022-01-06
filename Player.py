import multiprocessing as mp
import inspect
from time import perf_counter
import asyncio
import concurrent

from collections import deque

import _pickle as pickle
import networkx as nx
import numpy as np

from pyqtgraph.Qt.QtGui import *
from pyqtgraph.Qt.QtCore import *



from PyQtViz import edge_viewer
from util import mkprocess, get_creationtime, get_filenames
from GLNetworkItem import GLNetworkItem
from globals import basal_offset, save_pattern


def pickle_player(path='.', pattern=save_pattern, start_time=0, speedup=5.0, buffer_len=100, workers=2, **kw):
    def play():
        pass

    def setPosition():
        pass

    def setup_player(win):

        docker = QDockWidget(win)
        w = QWidget()
        playbackLayout = QHBoxLayout()

        layout = QVBoxLayout()
        
        playButton = QPushButton()
        playButton.setEnabled(False)
        playButton.setIcon(win.style().standardIcon(QStyle.SP_MediaPlay))
        playButton.clicked.connect(play)

        positionSlider = QSlider(Qt.Horizontal)
        positionSlider.setRange(0, 0)
        positionSlider.sliderMoved.connect(setPosition)

        playbackLayout.addWidget(playButton)
        playbackLayout.addWidget(positionSlider)



        layout.addLayout(playbackLayout)
        w.setLayout(layout)

        docker.setWidget(w)
        docker.setFloating(False)

        win.addDockWidget(Qt.BottomDockWidgetArea, docker)

    def load(entry):
        with open(entry[0], 'rb') as input:
            G=pickle.loads(input)
            q.append((G, entry[1])) 

    q = deque([],buffer_len)
    start_timestamp = get_creationtime(pattern.replace('*',str(start_time)), path=path)
    file_list = get_filenames(path=path, pattern=pattern, min_timestamp=start_timestamp)
    
    load(file_list[0])


    view = edge_viewer(G, window_callback=setup_player, refresh_rate=refresh_rate, **kw)
    refresh_interval = 1/refresh_rate

    def check_for_newfiles():
        pass
    
    curr_time = start_time
    dt=0
    now=perf_counter()
    last=now

    def loader():
        nonlocal curr_time, dt
        now=perf_counter()
        lag=0


        i=np.argmin(np.abs(file_list[:,1]-curr_time))
        load(file_list[i])
        while True:
            if speedup:
                if speedup>0:
                    sub_list=file_list[i:]
                elif speedup<0:
                    sub_list=file_list[:i]
                    pass
                for e in sub_list:
                    if speedup * e[1] >= speedup * (curr_time + dt * speedup):
                        load(e)
                        break
    

    def refresh():
            nonlocal curr_time, now, last, dt
            if len(q) and speedup:
                cont=True
                while cont:
                    G, time = q.pop_left()
                    play = speedup  and  speedup * time >= speedup * (curr_time + dt * speedup)
                    cont = len(q)  and not play

            if play:
                view(G, title=f't={time}')
                curr_time = time

            now = perf_counter()

            dt = now - last
            if not play:    
                curr_time += dt

            last = now
            sleep = refresh_interval - dt

            if sleep>0:
                time.sleep(sleep/2)

    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor()

    def run():

        loop.run_in_executor(executor, loader)
                
        while True:
            refresh()



    
    loop.run_in_executor(executor, check_for_newfiles)
    loop.run_in_executor(executor, run)
    





if __name__ == '__main__':


    pickle_player(attr='myosin', cell_edges_only=True, apical_only=True)
    
    while True:
        pass
