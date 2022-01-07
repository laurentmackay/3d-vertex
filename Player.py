import multiprocessing as mp
import inspect
import asyncio
import concurrent
import math
import itertools
import os

from time import perf_counter, sleep
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


def pickle_player(path=os.getcwd(), pattern=save_pattern, start_time=0, speedup=5.0, refresh_rate=60.0, buffer_len=100, workers=2, **kw):
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

    G=None
    latest = float('inf')
    def load(entry):
        nonlocal G, latest
        # print(f'loading {entry}')
        with open(entry[0], 'rb') as input:
            # print('inpout open')
            G=pickle.load(input)
            q.append((G, entry[1]))
            # print(f'loaded {entry[0]}')

        latest = entry[1]

    q = deque([],buffer_len)
    start_file = pattern.replace('*',str(start_time))
    start_timestamp = get_creationtime( start_file, path=path)
    
    load((start_file, start_time))


    view = edge_viewer(G, window_callback=setup_player, refresh_rate=refresh_rate, **kw)
    print(view)
    refresh_interval = 1/refresh_rate
    time_bounds = [float('inf'), float('-inf')]
    file_list = get_filenames(path=path, pattern=pattern, min_timestamp=start_timestamp)
    def check_for_newfiles():
        nonlocal time_bounds 
        while True:     
            # print('hihi') 
            get_filenames(path=path, pattern=pattern, min_timestamp=start_timestamp, extend=file_list)
            if time_bounds[0]>file_list[0][1]:
                time_bounds[0]=file_list[0][1]

            if time_bounds[1]<file_list[-1][1]:
                time_bounds[1]=file_list[-1][1]

            sleep(1)


    
    prev_time = start_time
    next_time = prev_time + refresh_interval*speedup
    dt=0
    now=perf_counter()
    prev_now=now
    i_load=0
    def loader():
        nonlocal prev_time, next_time, dt, latest, i_load
        print('loader runing')
        now=perf_counter()
        lag=0

        

        i_last = None

        while True:
            # print(f'hihi {i}')
           if (i_last is None) or i_last != i_load:
                load(file_list[i_load])
                i_last=i_load

            sleep(refresh_interval/2)


    t0=float('-inf')
    displayed=False
    Gdisp=G
  

    def load_next(time):
        nonlocal G, Gdisp, t0, i_load
        i_load=np.argmin(np.abs(np.array([e[1]-time for e in file_list])))
        print(i_load)
        load(file_list[i_load])
        Gdisp, t0 = q.popleft()
        

    def refresh():
            nonlocal prev_time, next_time, now, prev_now, dt, time_bounds, latest, t0, displayed, Gdisp
            now = perf_counter()
            curr_time = prev_time + (now - prev_now) * speedup
            search = len(q) and latest >= curr_time and curr_time >= next_time
            
            # if speedup and search and displayed:
            #     cont=True

            #     while cont:
            #         print(f'gonna pop {len(q)} {q.popleft()}')
            #         Gdisp, t0 = q.popleft()
            #         print(f'popped {t0}; curr_time {curr_time}; next_time {next_time}')
            #         display =  t0 >= next_time
            #         cont = len(q)  and  not display
            #         print(f'cont {cont}')
            #     displayed=False
            # else:
            #     print(f'not searching {len(q)} {latest} {curr_time} {next_time} {t0}')
                    
                    # print(cont)
            print(curr_time)
            if curr_time>=t0:
                print(f'qlen: {len(q)}')
                displayed=True
                # G, t = q.popleft()
                
                # print(f'siapl {t} {next_time}')
                view(Gdisp, title=f't={t0}')
                print('viewed')
                next_now = perf_counter()
                prev_time=curr_time+(next_now-now)*speedup
                now=next_now
                load_next(prev_time + refresh_interval * speedup)

                print(f'next_time {next_time}')
                # prev_time = t
                prev_now = now
            else:
                now = perf_counter()
            

            dt = now - prev_now
            
            # if not display:
            #     projected_time = curr_time + dt
            #     # print(f'project_time {projected_time} {time_bounds}')
            #     if projected_time<time_bounds[1] and projected_time>time_bounds[0]:
            #         # curr_time = projected_time
            #         # print(f'curr_time {curr_time}')

            # # print(f'curr_time: {curr_time}') 
            
            rem = refresh_interval - dt
            # print(f'sleeping : {rem}') 
            if rem>0:
                sleep(rem/2)
            else:
                sleep(refresh_interval/2)
            # print('wakey')

    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor()

    def run():

        while True:
            refresh()



    
    loop.run_in_executor(executor, check_for_newfiles)
    print('goona load')
    # loop.run_in_executor(executor, loader)
    print('that was done')  
    loop.run_in_executor(executor, run)
    
    # while True:
    #     sleep(1)




if __name__ == '__main__':


    pickle_player(attr='myosin', cell_edges_only=True, apical_only=True)
    
    # while True:
    #     pass
