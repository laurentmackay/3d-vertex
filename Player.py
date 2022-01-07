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

    pos=None
    def setPosition(a):
        nonlocal pos
        # print(a)
        print(pipe)
        pos=a

    def setter(b, key):
        def set(a):
            print({key:a})
            b.send({key:a})
        return set

    positionSlider = None
    playButton = None
    sliderPrec = 10**4
    tmr=None
    def setup_player(win, b):
        nonlocal tmr
        positionSlider = QSlider(Qt.Horizontal)
        playButton = QPushButton()

        docker = QDockWidget(win)
        w = QWidget()
        playbackLayout = QHBoxLayout()

        layout = QVBoxLayout()
        
        
        playButton.setEnabled(False)
        playButton.setIcon(win.style().standardIcon(QStyle.SP_MediaPlay))
        playButton.clicked.connect(play)

        
        positionSlider.setRange(0, sliderPrec)
        positionSlider.sliderMoved.connect(setter(b, 'pos'))

        playbackLayout.addWidget(playButton)
        playbackLayout.addWidget(positionSlider)



        layout.addLayout(playbackLayout)
        w.setLayout(layout)

        docker.setWidget(w)
        docker.setFloating(False)

        win.addDockWidget(Qt.BottomDockWidgetArea, docker)

        def listen():
            if b.poll():
                while b.poll():
                    msg=b.recv()
                    positionSlider.setSliderPosition(msg)


                    

        tmr = QTimer()
        tmr.timeout.connect(listen)
        tmr.setInterval(int(500/refresh_rate))
        tmr.start()

        print(f'done setting up {tmr}')

    


        


    start_file = pattern.replace('*',str(start_time))
    start_timestamp = get_creationtime( start_file, path=path)

    with open(start_file, 'rb') as input:
        G=pickle.load(input)
        t_G = start_time

    new=True

    view = edge_viewer(G, window_callback=setup_player, refresh_rate=refresh_rate, **kw)

    print(view)
    refresh_interval = 1/refresh_rate
    time_bounds = [start_time, float('-inf')]
    file_list = [(pattern.replace('*',str(start_time)), start_time)]


    def check_for_newfiles():
        nonlocal time_bounds, positionSlider

        get_filenames(path=path, pattern=pattern, min_timestamp=start_timestamp, extend=file_list)
        latest_timestamp = get_creationtime(file_list[-1][0])
        time_bounds[1]=file_list[-1][1]

        while True:     
            sleep(0.1)
            
            get_filenames(path=path, pattern=pattern, min_timestamp=latest_timestamp, extend=file_list)
            latest_timestamp = get_creationtime(file_list[-1][0])

            if time_bounds[0]>file_list[0][1]:
                time_bounds[0]=file_list[0][1]

            if time_bounds[1]<file_list[-1][1]:
                time_bounds[1]=file_list[-1][1]
            

            


    

    i_load=0
    i_loaded = None
    prev_disp_time = start_time
    next_disp_time = start_time

    def loader():
        nonlocal  i_load, i_loaded, G, t_G, new, next_disp_time   

        while True:

            for i_load, e in enumerate(file_list):
                if e[1]>=next_disp_time:
                    break

            if (i_loaded is None) or i_loaded != i_load:
                try:
                    file, t = file_list[i_load]
                    with open(file, 'rb') as input:
                        G=pickle.load(input)
                        t_G = t

                        i_loaded=i_load
                        new = True
                except:
                    pass
            sleep(refresh_interval)

        



           

    counter=perf_counter()
    prev_counter=counter

    def refresh():
            nonlocal prev_disp_time, next_disp_time,  counter, prev_counter, t_G, G, i_loaded, new, positionSlider, pos
            counter = perf_counter()
            curr_time = prev_disp_time + (counter - prev_counter) * speedup

            if curr_time>=t_G and new:

                view(G, title=f't={t_G}')
                #update times
                prev_counter =  perf_counter()
                curr_time += (prev_counter-counter)*speedup
                prev_disp_time = curr_time
                
                #update states
                new = False
                requesting_load = True
                next_disp_time = curr_time + refresh_interval * speedup
              
            else:
                now = perf_counter()

            if i_loaded is not None:
                rem = (file_list[i_loaded][1]-curr_time)/speedup
                # print(f'rem {rem}')
            else:
                rem = refresh_interval

            timespan = time_bounds[1] - time_bounds[0]
            idx = int(sliderPrec*(curr_time-time_bounds[0])/(timespan))
            view.pipe.send(idx)

            if rem>0:
                sleep(refresh_interval/2)
            else:
                sleep(0.1)


            # print('wakey')

    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor()

    def run():
        while True:
            refresh()



    
    loop.run_in_executor(executor, check_for_newfiles)
    print('goona load')
    loop.run_in_executor(executor, loader)
    print('that was done')  
    loop.run_in_executor(executor, run)
    
    # while True:
    #     sleep(1)




if __name__ == '__main__':


    pickle_player(attr='myosin', cell_edges_only=True, apical_only=True)
    
    # while True:
    #     pass
