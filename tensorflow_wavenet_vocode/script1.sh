alias python=/usr/bin/python
CUDA_VISIBLE_DEVICES=0 python train.py --batch_size=50 \
--data_dir=/posefs1b/Users/hanbyulj/ssp-data/data_face_multi3_rootOnly_dtdr_rel_2d/haggling3/training_std \
--checkpoint_every=1000 --num_steps=65000 --sample_size=200 --learning_rate=1e-3 --optimizer=adam --histograms=False \
--logdir_root=/posefs1b/Users/luna/wavenet --wavenet_params=./wavenet_params1.json --epsilon=1e-7 

