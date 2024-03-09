# https://medium.com/@niveditha.itengineer/learn-how-to-setup-portaudio-and-pyaudio-in-ubuntu-to-play-with-speech-recognition-8d2fff660e94


import numpy as np
import scipy.signal
from scipy.special import softmax
import timeit
import python_speech_features
import os
import json
#import RPi.GPIO as GPIO
#from kws_streaming.data import input_data
from kws_streaming import data
from datetime import datetime
import pandas as pd

from tflite_runtime.interpreter import Interpreter
#import tensorflow as tf
import time as timelib
import sounddevice as sd

# Parameters
#debug_time = 1
#debug_acc = 0
#led_pin = 8
word_threshold = 0.9

sample_rate = 16000

num_channels = 1
model_path = 'models3_30k/svdf/tflite_stream_state_external/stream_state_external.tflite'


def get_menufilter():
    global index_to_label, currentmenu, menufilter

    menufilter = np.zeros(len(index_to_label), dtype=np.float32)
    menufilter[int(label_to_index[rootmenu['kw']])] = 1
    for i in currentmenu['submenu']:
        menufilter[int(label_to_index[i['kw']])] = 1
    return menufilter


def sd_callback(rec, frames, time, status):   
    
    #global sumseconds, sumrecords
    global predict, probat, index_to_label, currentmenu, menufilter


    # audio rec shape is : (frames, channels)
    rec = np.reshape(rec, (1,-1))

    # set input audio data (by default input data at index 0)
    interpreter.set_tensor(input_details[0]['index'], rec)

    # set input states (index 1...)
    for s in range(1, len(input_details)):
        interpreter.set_tensor(input_details[s]['index'], inputs[s])

    # run inference
    interpreter.invoke()

    # get output: classification
    out_tflite = interpreter.get_tensor(output_details[0]['index'])
    #print(start / 16000.0, np.argmax(out_tflite), np.max(out_tflite))

    # get output states and set it back to input states
    # which will be fed in the next inference cycle
    for s in range(1, len(input_details)):
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
        inputs[s] = interpreter.get_tensor(output_details[s]['index'])

    
    probas = softmax(out_tflite)

    proba = np.max(probas)
    #if (proba > word_threshold):
    #    out_tflite_argmax = np.argmax(out_tflite)
    #    if out_tflite_argmax > 1:
    #        label_found = index_to_label[str(out_tflite_argmax)]
    #        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), label_found, proba)

    # set to zero all probability on not expected key words
    out_tflitefiltered = out_tflite * menufilter
    probas_filtered = probas * menufilter

    out_tflite_argmax_filtered = np.argmax(out_tflitefiltered)

    
    proba_filtered = np.max(probas_filtered)
    if (proba_filtered > word_threshold):
        label_found = index_to_label[str(out_tflite_argmax_filtered)]
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S ===="), label_found, proba)
        
        # moving in menu
        if rootmenu['kw'] == label_found:
           currentmenu = rootmenu
        else:
            for submenu in currentmenu["submenu"]:
                if submenu['kw'] == label_found:
                    foundmenu = submenu
                    break
            currentmenu = foundmenu

            if currentmenu.get("submenu") is None:
                print(datetime.now().strftime("%d/%m/%Y %H:%M:%S ==== ACTION"), label_found, proba)
                currentmenu = rootmenu

        menufilter = get_menufilter()



    



# GPIO 
#GPIO.setwarnings(False)
#GPIO.setmode(GPIO.BOARD)
#GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW)

current_dir = os.getcwd()
DATA_PATH = os.path.join(current_dir, "data2/")
MODELS_PATH = current_dir

file_name='models3_30k'
train_dir = os.path.join(MODELS_PATH, file_name, 'svdf')

# below is another way of reading flags - through json
with open(os.path.join(train_dir, 'flags.json'), 'r') as fd:
   flags_json = json.load(fd)

class DictStruct(object):
   def __init__(self, **entries):
     self.__dict__.update(entries)

flags = DictStruct(**flags_json)

print('flags:',flags)


with open(os.path.join(train_dir, 'index_to_label.json'), 'r') as fd2:
   index_to_label = json.load(fd2)

label_to_index = {v: k for k, v in index_to_label.items()}
print('label_to_index', label_to_index)

with open(os.path.join(current_dir, 'menu.json'), 'r') as fd3:
   rootmenu = json.load(fd3)

currentmenu = rootmenu
menufilter = get_menufilter()


#interpreter = tf.lite.Interpreter(model_path,num_threads=20)
interpreter = Interpreter(model_path,num_threads=1)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('---input_details:', input_details)

inputs = []
for s in range(len(input_details)):
  inputs.append(np.zeros(input_details[s]['shape'], dtype=np.float32))

input_details[0]['shape']


reset_state = True


# before processing new test sequence we can reset model state
# if we reset model state then it is not real streaming mode
if reset_state:
  for s in range(len(input_details)):
    print(input_details[s]['shape'])
    inputs[s] = np.zeros(input_details[s]['shape'], dtype=np.float32)


print('---reset_state done')


sumseconds=0
sumrecords=0
MAX_RECORDS = 5000



print('---launching loop')

try:
    sumrecords=0
    print('channels:',num_channels, 'samplerate:',sample_rate, 'blocksize:', flags.window_stride_samples)
    with sd.InputStream(channels=num_channels,
                        samplerate=sample_rate,
                        blocksize=flags.window_stride_samples,
                        callback=sd_callback):
        while True:
            pass

except KeyboardInterrupt:
    pass

print('---loop closed')
   


