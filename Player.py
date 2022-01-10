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

import pyqtgraph as pg
from pyqtgraph.Qt.QtGui import *
from pyqtgraph.Qt.QtCore import *



from PyQtViz import edge_viewer
from util import mkprocess, get_creationtime, get_filenames
from GLNetworkItem import GLNetworkItem
from globals import basal_offset, save_pattern


def pickle_player(path=os.getcwd(), pattern=save_pattern, start_time=0, speedup=5.0, refresh_rate=60.0, buffer_len=100, workers=2, **kw):


    start_file = pattern.replace('*',str(start_time))
    start_timestamp = get_creationtime( start_file, path=path)


    refresh_interval = 1/refresh_rate
    time_bounds = [start_time, float('-inf')]
    file_list = [(pattern.replace('*',str(start_time)), start_time)]

    i_load=0
    i_loaded = None
    prev_disp_time = start_time
    next_disp_time = start_time
    curr_time = start_time

    counter=perf_counter()
    prev_counter=counter

    slider_ticks = 10**4
    positionSlider = None

    def setPosition(a):
        nonlocal curr_time, next_disp_time, new
        curr_time = a/slider_ticks*(time_bounds[1]-time_bounds[0])
        next_disp_time = curr_time + refresh_interval * speedup

    advance = True
    def freeze():
        nonlocal advance
        advance = False

    def unfreeze():
        nonlocal advance, new
        advance = True
        new = True


    def setup_player(win):
        nonlocal positionSlider
        positionSlider = QSlider(Qt.Horizontal)
        playButton = QPushButton()

        docker = QDockWidget(win)
        w = QWidget()
        playbackLayout = QHBoxLayout()

        layout = QVBoxLayout()
        
        
        playButton.setEnabled(False)
        playButton.setIcon(win.style().standardIcon(QStyle.SP_MediaPlay))
        # playButton.clicked.connect(play)

        
        positionSlider.setRange(0, slider_ticks)
        positionSlider.sliderMoved.connect(setPosition)
        positionSlider.sliderPressed.connect(freeze)
        positionSlider.sliderReleased.connect(unfreeze)

        playbackLayout.addWidget(playButton)
        playbackLayout.addWidget(positionSlider)



        layout.addLayout(playbackLayout)
        w.setLayout(layout)

        docker.setWidget(w)
        docker.setFloating(False)

        win.addDockWidget(Qt.BottomDockWidgetArea, docker)




    def check_for_newfiles():
        nonlocal time_bounds

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
                        # print(f'loading {t}')
                        i_loaded=i_load
                        new = True
                except:
                    pass
            sleep(refresh_interval/2)

        





    def refresh():
            nonlocal prev_disp_time, curr_time, next_disp_time,  counter, prev_counter, t_G, G, i_loaded, new, advance
            counter = perf_counter()
            if advance:
                curr_time += (counter - prev_counter) * speedup

            curr_time=max(min(curr_time, time_bounds[1]),time_bounds[0])
            dt=curr_time-prev_disp_time

            # print(f'current time {curr_time} {dt>=t_G-prev_disp_time}')

            if (not advance or dt>=t_G-prev_disp_time) and new:

                view(G, title=f't={t_G}')
                #update times
                prev_counter =  perf_counter()
                if advance:
                    curr_time += (prev_counter-counter)*speedup

                curr_time=max(min(curr_time,time_bounds[1]),time_bounds[0])
                prev_disp_time = curr_time
                
                #update states
                new = False
                next_disp_time = curr_time + refresh_interval * speedup
              
            else:

                prev_counter = perf_counter()

            if i_loaded is not None:
                rem = (file_list[i_loaded][1]-curr_time)/speedup
            else:
                rem = refresh_interval

            timespan = time_bounds[1] - time_bounds[0]
            idx = int(slider_ticks*(curr_time-time_bounds[0])/(timespan))
            positionSlider.setSliderPosition(idx)

            if rem>0:
                sleep(refresh_interval/2)
            else:
                sleep(0.1)


    with open(start_file, 'rb') as input:
        G=pickle.load(input)
        t_G = start_time

    new=True
    
    view = edge_viewer(G, window_callback=setup_player, refresh_rate=refresh_rate, parallel=False, **kw)

    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor()

    def run():
        while True:
            refresh()

    def exit():
        for f in futures:
            f.cancel()
        executor.shutdown(wait=False)
        concurrent.futures.thread._threads_queues.clear()
        loop.stop()


    futures=[loop.run_in_executor(executor,func) for func in (check_for_newfiles, loader, run)]
    
    app = QApplication.instance()
    app.aboutToQuit.connect(exit) 

    pg.exec()




if __name__ == '__main__':


    pickle_player(attr='myosin', cell_edges_only=True, apical_only=True)
    
    # while True:
    #     pass
