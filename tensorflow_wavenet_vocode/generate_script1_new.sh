#!/bin/bash
# pass in two arguments, arg1: timestamp of model, arg2: wavenet_params file used by the model. Fix sample/window size to 100/100 by now. 
modeldir='/media/posefs1b/Users/luna/wavenet/train/'
datadir='/media/posefs1b/Users/hanbyulj/ssp-data/data_face_multi3_rootOnly_dtdr_rel_2d/haggling3/testing_std_shift100_withFrameNumber/'
generatedir='/media/posefs1b/Users/luna/wavenet/generate_skeleton_window200sample100/'
scenelist=`ls $datadir`
timestamp=$1
params=$2
ckpt='/model.ckpt-4000'
for dir in $scenelist
do
	scenedir=$datadir$dir
	filelist=`ls $scenedir`
	for file in $filelist
	do
        CUDA_VISIBLE_DEVICES=3 python generate.py --wavenet_params=$params --samples=100 --window=200 \
        --skeleton_out_path=$generatedir --motion_seed="$scenedir/$file" $modeldir$timestamp$ckpt
    done
done



