#!/bin/bash
# pass in two arguments, arg1: timestamp of model, arg2: wavenet_params file used by the model. Fix sample/window size to 100/100 by now. 
modeldir='/posefs1b/Users/luna/wavenet/train/'
datadir='/posefs1b/Users/hanbyulj/ssp-data/data_multiple_2_rotDec_deltaT_tree_rootOnly_noRef/haggling3_dt_keepDist/testing_std_shift100/'
generatedir='/posefs1b/Users/luna/wavenet/generate_skeleton_window200sample100/'
dirlist=`ls $datadir`
timestamp=$1
params=$2
ckpt='/model.ckpt-64999'
for dir in $dirlist
do
	posedir=$datadir$dir
	#python generate_skeleton.py --wavenet_params=wavenet_params1.json --samples=100 --window=500 --skeleton_out_path=wavenet_generated.mat --motion_seed=$posedir /home/luna/ssp/model/tensorflow-wavenet-ssp/logdir-server6/train/2017-01-15T07-10-00/model.ckpt-64999.data-00000-of-00001
    CUDA_VISIBLE_DEVICES=0 python generate.py --wavenet_params=$params --samples=100 --window=200 \
    --skeleton_out_path=$generatedir --motion_seed=$posedir $modeldir$timestamp$ckpt \
    --gc_channels=1
done



