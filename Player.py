from curses import window
import multiprocessing as mp
import inspect
import asyncio
import concurrent
import math
import itertools
import os

from time import perf_counter, sleep
from collections import deque

import pickle
import time 
import networkx as nx
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt.QtGui import *
from pyqtgraph.Qt.QtCore import *



from PyQtViz import edge_viewer
from util import mkprocess, get_creationtime, get_filenames
from GLNetworkItem import GLNetworkItem
from globals import basal_offset, save_pattern

# class ClickSlider(QSlider):
#     """A slider with a signal that emits its position when it is pressed. Created to get around the slider only updating when the handle is dragged, but not when a new position is clicked"""

#     sliderPressedWithValue = Signal(int)

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.sliderPressed.connect(self.on_slider_pressed)

#     def on_slider_pressed(self):
#         """emits a more descriptive signal when pressed (with slider value during the press event)"""
#         self.sliderPressedWithValue.emit(self.value())

class ClickSlider(QSlider):
    def mousePressEvent(self, event):
        super(ClickSlider, self).mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            val = self.pixelPosToRangeValue(event.pos())
            self.setValue(val)
            self.sliderMoved.emit(val)

    def pixelPosToRangeValue(self, pos):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        gr = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        sr = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)

        if self.orientation() == Qt.Horizontal:
            sliderLength = sr.width()
            sliderMin = gr.x()
            sliderMax = gr.right() - sliderLength + 1
        else:
            sliderLength = sr.height()
            sliderMin = gr.y()
            sliderMax = gr.bottom() - sliderLength + 1;
        pr = pos - sr.center() + sr.topLeft()
        p = pr.x() if self.orientation() == Qt.Horizontal else pr.y()
        return QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), p - sliderMin,
                                               sliderMax - sliderMin, opt.upsideDown)



def pickle_player(path=os.getcwd(), pattern=save_pattern, start_time=0, speedup=5.0, refresh_rate=60.0, parallel=False, workers=2, **kw):


    start_file = pattern.replace('*',str(start_time))
    start_timestamp = get_creationtime( start_file, path=path)

    # start_file = os.path.join(path, start_file)
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
    playButton = None
    window = None

    new = True
    playing = True

    view=None
    t_G=None
    G=None

    def setPosition(a):
        nonlocal curr_time, next_disp_time, new
        curr_time = a/slider_ticks*(time_bounds[1]-time_bounds[0])
        next_disp_time = curr_time + refresh_interval * speedup

    
    def freeze():
        nonlocal playing
        playing = False

    def unfreeze():
        nonlocal playing, new
        playing = True
        new = True

    def pushPlay():
        nonlocal window, playButton
        nonlocal playing
        if playing:
            freeze()
            playButton.setIcon(window.style().standardIcon(QStyle.SP_MediaPause))
        else:
            unfreeze()
            playButton.setIcon(window.style().standardIcon(QStyle.SP_MediaPlay))


    def setup_player(win):
        nonlocal positionSlider, playButton, window

        window = win
        positionSlider = ClickSlider(Qt.Horizontal)
        playButton = QPushButton()

        docker = QDockWidget(win)
        w = QWidget()
        playbackLayout = QHBoxLayout()

        layout = QVBoxLayout()
        
        
        playButton.setEnabled(True)
        playButton.setIcon(win.style().standardIcon(QStyle.SP_MediaPlay))
        playButton.clicked.connect(pushPlay)

        
        positionSlider.setRange(0, slider_ticks)
        positionSlider.sliderMoved.connect(setPosition)
        # positionSlider.sliderPressedWithValue.connect(setPosition)
        positionSlider.sliderPressed.connect(freeze)
        positionSlider.sliderReleased.connect(unfreeze)


        playbackLayout.addWidget(playButton)
        playbackLayout.addWidget(positionSlider)



        layout.addLayout(playbackLayout)
        w.setLayout(layout)

        docker.setWidget(w)
        docker.setFloating(False)

        win.addDockWidget(Qt.BottomDockWidgetArea, docker)


    def syncSlider():
        nonlocal curr_time, time_bounds
        
        timespan = time_bounds[1] - time_bounds[0]
        if timespan:
            idx = int(slider_ticks*(curr_time-time_bounds[0])/(timespan))
            positionSlider.setSliderPosition(idx)

    def check_for_newfiles():
        nonlocal time_bounds

        get_filenames(path=path, pattern=pattern, min_timestamp=start_timestamp, extend=file_list)
        latest_timestamp = get_creationtime(os.path.join(path,file_list[-1][0]))
        time_bounds[1]=file_list[-1][1]

        while True:     
            sleep(0.1)
            

            get_filenames(path=path, pattern=pattern, min_timestamp=latest_timestamp, extend=file_list)
            latest_timestamp = get_creationtime(os.path.join(path,file_list[-1][0]))

            changed = False
            if time_bounds[0]>file_list[0][1]:
                time_bounds[0]=file_list[0][1]
                changed = True

            if time_bounds[1]<file_list[-1][1]:
                time_bounds[1]=file_list[-1][1]
                changed = True

            if changed:
                syncSlider()
            

            


    



    def loader():
        nonlocal  i_load, i_loaded, G, t_G, new, next_disp_time   

        while True:

            for i_load, e in enumerate(file_list):
                if e[1]>=next_disp_time:
                    break

            if (i_loaded is None) or i_loaded != i_load:
                try:
                    file, t = file_list[i_load]
                    with open(os.path.join(path,file), 'rb') as input:
                        try:
                            G=pickle.load(input)
                            t_G = t
                            # print(f'loading {t}')
                            i_loaded=i_load
                            new = True
                        except:
                            pass
                except:
                    pass
            sleep(refresh_interval/2)

        




    

    def refresh():
            nonlocal prev_disp_time, curr_time, next_disp_time,  counter, prev_counter, t_G, G, i_loaded, new, playing, view
            while t_G is None:
                sleep(refresh_interval/2)

            counter = perf_counter()
            if playing:
                curr_time += (counter - prev_counter) * speedup

            curr_time=max(min(curr_time, time_bounds[1]),time_bounds[0])
            dt=curr_time-prev_disp_time

            # print(f'current time {curr_time} {dt>=t_G-prev_disp_time}')

            if (not playing or dt>=t_G-prev_disp_time) and new:

                view(G, title=f't={t_G}')
                #update times
                prev_counter =  perf_counter()
                if playing:
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

            syncSlider()

            if rem>0:
                sleep(rem/2)
            else:
                sleep(refresh_interval/2)

    def start():
        nonlocal view, counter, prev_counter
        with open(os.path.join(path, start_file), 'rb') as input:
            G=pickle.load(input)
            t_G = start_time
        

        

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

        counter=perf_counter()
        prev_counter=counter

        futures=[ loop.run_in_executor(executor,f) for f in (check_for_newfiles, loader, run) ]
        
        app = QApplication.instance()
        app.aboutToQuit.connect(exit) 

        pg.exec()

    if parallel:
        proc=mkprocess(start)
        proc.start()
    else:
        start()



if __name__ == '__main__':


    pickle_player(attr='myosin', cell_edges_only=True, apical_only=True)
    
    # while True:
    #     pass
