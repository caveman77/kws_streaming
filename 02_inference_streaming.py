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




def sd_callback(rec, frames, time, status):   
    
    #global sumseconds, sumrecords
    global index, times, MAX_RECORDS, predict, probat, index_to_label

    if index<MAX_RECORDS-1:
        index = index+1

        #tstart = timelib.time()
        times[0, index] = timelib.time()

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

    out_tflite_argmax = np.argmax(out_tflite)

    probas = softmax(out_tflite)
    proba = np.max(probas)
    if (out_tflite_argmax>1) and (proba > word_threshold):
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), index_to_label[str(out_tflite_argmax)], proba)
        #print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), out_tflite_argmax, proba)

    if index<MAX_RECORDS:
        times[1, index] = timelib.time()
        predict[index] = out_tflite_argmax
        probat[index] = proba

    



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

# prepare mapping of index to word
#audio_processor = data.input_data.AudioProcessor(flags)
#index_to_label = {}
# labels used for training
#for word in audio_processor.word_to_index.keys():
#  if audio_processor.word_to_index[word] == data.input_data.SILENCE_INDEX:
#    index_to_label[audio_processor.word_to_index[word]] = data.input_data.SILENCE_LABEL
#  elif audio_processor.word_to_index[word] == data.input_data.UNKNOWN_WORD_INDEX:
#    index_to_label[audio_processor.word_to_index[word]] = data.input_data.UNKNOWN_WORD_LABEL
#  else:
#    index_to_label[audio_processor.word_to_index[word]] = word

with open(os.path.join(train_dir, 'index_to_label.json'), 'r') as fd2:
   index_to_label = json.load(fd2)
   
print('---index_to_label:',index_to_label)
print('23-1:', index_to_label['23'])
print('23-2:', index_to_label[str(23)])


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
index=-1
times = np.empty([2, MAX_RECORDS], dtype = float)
predict = np.empty([ MAX_RECORDS], dtype = int)
probat = np.empty([ MAX_RECORDS], dtype = float)

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
   
times[0,:] = times[0,:] * -1
delays = np.sum(times[:,0:index], axis=0)

df = pd.DataFrame(delays, columns=['duration'])
df['probability'] = probat[0:index]
df['predict'] = predict[0:index]

df.to_csv('timing.csv')

