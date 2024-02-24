# kws_streaming

This is based on google research repository that can be find here : https://github.com/google-research/google-research/blob/master/kws_streaming/README.md

This clone is not the latest and is close to the 2020 release. Few modifications have been done afterward.

This is based on Python 3.7 and TF 2.4

requirements.txt file is the one used to get the packages. The one in the subfolder is the original one and is to be ignored.

KWS_PATH=$PWD
CMD_TRAIN="python -m kws_streaming.train.model_train_eval"
DATA_PATH=$KWS_PATH/data2
MODELS_PATH=$KWS_PATH/models3_30k

$CMD_TRAIN --data_url '' --wanted_words 'backward,bed,bird,cat,dog,down,eight,five,follow,forward,four,go,happy,house,learn,left,marvin,nine,no,off,on,one,right,seven,sheila,six,stop,three,tree,two,up,visual,wow,yes,zero' --data_dir $DATA_PATH/ --train_dir $MODELS_PATH/svdf/ --mel_upper_edge_hertz 7000 --how_many_training_steps 20000,20000,20000,20000 --learning_rate 0.001,0.0005,0.0001,0.00002 --window_size_ms 40.0 --window_stride_ms 20.0 --mel_num_bins 80 --dct_num_features 40 --resample 0.15 --time_shift_ms 100 --feature_type 'mfcc_op' --preprocess 'raw' --train 1 --lr_schedule 'exp' svdf --svdf_memory_size 4,10,10,10,10,10 --svdf_units1 16,32,32,32,64,128 --svdf_act "'relu','relu','relu','relu','relu','relu'" --svdf_units2 40,40,64,64,64,-1 --svdf_dropout 0.0,0.0,0.0,0.0,0.0,0.0 --svdf_pad 0 --dropout1 0.0 --units2 '' --act2 ''


