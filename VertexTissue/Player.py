import asyncio
import concurrent

import os

from time import perf_counter, sleep
from multiprocessing import Pipe

import pickle


import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt.QtWidgets import  QLabel, QSlider, QStyleOptionSlider, QPushButton, QApplication,  QStyle,  QDockWidget, QWidget, QHBoxLayout, QVBoxLayout, QMainWindow
from pyqtgraph.Qt.QtCore import Qt, QTimer, QCoreApplication

from VertexTissue.util import finder



from .PyQtViz import edge_viewer
from ResearchTools.Multiprocessing import Process
from ResearchTools.Iterable import first_item
from ResearchTools.Dict import last_dict_key
from ResearchTools.Filesystem import get_creationtime, get_filenames
from .globals import save_pattern

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




def pickle_player(path=os.getcwd(), pattern=save_pattern, file_list=None, start_time=None, speedup=5.0, refresh_rate=60.0,
                   parallel=False, save_dict=None, pre_process=None, check_timestamp=True,  save_and_quit=False, **kw):

    if file_list is None:

        single_pickle = pattern.find('*') == -1 or save_dict is not None
        start_file = pattern

        if not (single_pickle):
            if start_time is None:
                file_list=get_filenames(path=path, pattern=pattern, min_timestamp=0)
                if len(file_list)==0:
                    print(f'No file mathcing the pattern {pattern} was found in {path}' )
                    return 
                start_time=file_list[0][1]
            else:
                file_list=[]

            start_file = start_file.replace('*',str(start_time))
            start_timestamp = get_creationtime( start_file, path=path)
        else:
            file_list=[]
            
    if save_dict is not None and save_dict.keys():
        start_time = save_dict.keys().__iter__().__next__()

    # start_file = os.path.join(path, start_file)
    refresh_interval = 1/refresh_rate
    time_bounds = [start_time, float('-inf')]
    

    i_load=0
    i_loaded = None
    prev_disp_time = start_time
    next_disp_time = start_time
    curr_time = start_time

    counter=perf_counter()
    prev_counter=counter

    slider_ticks = 10**4
    positionSlider = None
    positionLabel = None
    playButton = None
    window = None

    new = True
    playing = speedup>0

    view=None
    t_G=None
    G=None
    keys = None
    def setPosition(a):
        nonlocal curr_time, next_disp_time, positionLabel
        curr_time = time_bounds[0] + a/slider_ticks*(time_bounds[1]-time_bounds[0])
        positionLabel.setText(f"{curr_time:4.2f}/{time_bounds[1]:4.2f}")
        next_disp_time = curr_time + refresh_interval * speedup


        

    def sliderGrabbed():
        nonlocal playButton
        if playButton.isChecked():
            print('freezing')
            freeze()

    def sliderReleased():
        nonlocal playButton
        if playButton.isChecked():
            freeze()

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
            playButton.setChecked(False)
            freeze()
            playButton.setIcon(window.style().standardIcon(QStyle.SP_MediaPause))
        else:
            playButton.setChecked(True)
            unfreeze()
            playButton.setIcon(window.style().standardIcon(QStyle.SP_MediaPlay))

    def syncSlider():
        nonlocal curr_time, time_bounds, positionLabel
        
        timespan = time_bounds[1] - time_bounds[0]
        if timespan:
            idx = int(slider_ticks*(curr_time-time_bounds[0])/(timespan))
            positionSlider.setSliderPosition(idx)
            positionLabel.setText(f"{curr_time:4.2f}/{time_bounds[1]:4.2f}")

    def setup_player(win):
        nonlocal positionSlider, positionLabel, playButton, window

        window = win

        positionSlider = ClickSlider(Qt.Horizontal)
        playButton = QPushButton()
        playButton.setCheckable(True)

        docker = QDockWidget(win)
        w = QWidget()
        playbackLayout = QHBoxLayout()

        layout = QVBoxLayout()
        
        if playing:
            playButton.setEnabled(True)
            playButton.setIcon(win.style().standardIcon(QStyle.SP_MediaPlay))
            playButton.clicked.connect(pushPlay)
            playButton.setChecked(True)
        else:
            playButton.setEnabled(False)
            playButton.setIcon(win.style().standardIcon(QStyle.SP_MediaPause))

        
        positionSlider.setRange(0, slider_ticks)
        positionSlider.sliderMoved.connect(setPosition)
        # positionSlider.sliderPressedWithValue.connect(setPosition)
        positionSlider.sliderPressed.connect(sliderGrabbed)
        positionSlider.sliderReleased.connect(sliderReleased)

        positionLabel = QLabel("/")

        playbackLayout.addWidget(playButton)
        playbackLayout.addWidget(positionSlider)
        playbackLayout.addWidget(positionLabel)



        layout.addLayout(playbackLayout)
        w.setLayout(layout)

        docker.setWidget(w)
        docker.setFloating(False)

        win.addDockWidget(Qt.BottomDockWidgetArea, docker)




    def check_timebounds_changed():
        nonlocal time_bounds, save_dict, keys, file_list, curr_time, next_disp_time

        if len(file_list)==0:
            file_list = [(pattern.replace('*',str(start_time)), start_time)]

        multi_file=not (single_pickle or save_dict)

        while True:     
            sleep(0.05)
            
            if multi_file:
                get_filenames(path=path, pattern=pattern, min_timestamp=latest_timestamp if check_timestamp else 0, extend=file_list)
                latest_timestamp = get_creationtime(os.path.join(path,file_list[-1][0]))
                new_min, new_min = file_list[0][1], file_list[-1][1]
            else:
                keys = np.array(list(save_dict.keys()))             
                new_min, new_max = min(keys), max(keys)

            changed = False
            if time_bounds[1]<new_max:

                if curr_time==time_bounds[1]:
                    curr_time=new_max
                    next_disp_time = curr_time + refresh_interval * speedup

                time_bounds[1]=new_max
                changed=True


            if time_bounds[0]>new_min:

                time_bounds[0]=new_min
                changed=True

            if changed:
                
                syncSlider()





    

    keys=None

    def loader():
        nonlocal  i_load, i_loaded, G, t_G, new, next_disp_time  , keys, save_dict, file_list
        multi_file = not  (single_pickle or save_dict)
        while True:

            if multi_file :
                for i_load, e in enumerate(file_list):
                    if e[1]>=next_disp_time:
                        break
            else:
                for i_load, t in enumerate(keys):
                    if t>=next_disp_time:
                        break

            if (i_loaded is None) or i_loaded != i_load:
            #     times = [e[1] for _, e in enumerate(file_list)]

                try:
                    if multi_file:
                        file, t = file_list[i_load]
                        with open(os.path.join(path,file), 'rb') as input:
                            try:
                                if pre_process is not None:
                                    G=pre_process(pickle.load(input))
                                else:
                                    G=pickle.load(input)
                                t_G = t

                                i_loaded=i_load
                                new = True
                            except:
                                pass
                    else:

                        t_G=keys[i_load]
                        G=save_dict[t_G]
                        if pre_process is not None:
                            G=pre_process(G)
                        i_loaded=i_load
                        new = True

                except:
                    pass
            sleep(refresh_interval/4)

        




    

    def refresh():
            nonlocal prev_disp_time, curr_time, next_disp_time,  counter, prev_counter, t_G, G, i_loaded, new, playing, view
            while t_G is None:
                sleep(refresh_interval/2)

            counter = perf_counter()
            if playing:
                curr_time += (counter - prev_counter) * speedup

            curr_time=max(min(curr_time, time_bounds[1]),time_bounds[0])
            dt=curr_time-prev_disp_time

            if (not playing or dt>=t_G-prev_disp_time) and new:


                view(G, title=f't={t_G:.2f}')
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

            if i_loaded is not None and speedup:
                if not (single_pickle or save_dict):
                    rem = (file_list[i_loaded][1]-curr_time)/speedup
                else:
                    rem = (keys[i_loaded]-curr_time)/speedup
            else:
                rem = refresh_interval

            # print(f'syncing slider: {curr_time}')
            syncSlider()

            if rem>0:
                sleep(refresh_interval/2)
            else:
                sleep(refresh_interval/2)

    def start():
        def start_base():
            nonlocal view, counter, prev_counter, save_dict, keys, start_time, prev_disp_time, next_disp_time, curr_time, time_bounds
            if not single_pickle:
                with open(os.path.join(path, start_file), 'rb') as input:
                    G=pickle.load(input)
                    t_G = start_time
                # if not save_dict:
                #     save_dict=G

            else:
                if save_dict is None:
                    with open(os.path.join(path, start_file), 'rb') as input:
                        save_dict=pickle.load(input)

                if start_time is None:
                    start_time = first_item(save_dict)
                elif start_time==-1:
                    start_time = last_dict_key(save_dict)

                prev_disp_time = start_time
                next_disp_time = start_time
                curr_time = start_time
                time_bounds = [start_time, float('-inf')]

                keys = np.array(list(save_dict.keys()))
                start_ind = np.argmin(np.abs(keys-start_time))
                t_G=keys[start_ind]
                G=save_dict[t_G]
                

            if pre_process is not None:
                G=pre_process(G)
            view = edge_viewer(G, window_callback=setup_player, refresh_rate=refresh_rate, parallel=False, save_and_quit=save_and_quit, **kw)



            loop = asyncio.get_event_loop()
            executor = concurrent.futures.ThreadPoolExecutor()



            def exit():
                print("player exit func hase been called")
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=False)
                concurrent.futures.thread._threads_queues.clear()
                loop.stop()

            counter=perf_counter()
            prev_counter=counter

            futures=[ loop.run_in_executor(executor,check_timebounds_changed), loop.run_in_executor(executor, loader)] 
            

            app = QApplication.instance()
            app.aboutToQuit.connect(exit) 

            refresher = QTimer()
            refresher.timeout.connect(refresh)

            refresher.setInterval(int(500/refresh_rate))
            refresher.start()
            return refresher

        if parallel:
            def start_in_parallel(b):
                foo=start_base()
                def listen():
                    nonlocal save_dict

                    if b.poll():
                        while b.poll(): #empty the pipe if there has been some accumulation
                            (t_new, G_new)=b.recv()
                            save_dict[t_new]=G_new

                listener = QTimer()
                listener.timeout.connect(listen)
                
                listener.setInterval(int(500))
                listener.start()

                pg.exec()

            return start_in_parallel
        else:
            foo=start_base()
            pg.exec()
        
            

    if parallel:
        a,b = Pipe()
        proc=Process(start(), args=(b,), daemon=True)
        proc.start()
        return a
    else:
        start()

