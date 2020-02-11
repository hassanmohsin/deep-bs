# original kdeep 
python train.py --dataroot ./dataset/refined-set \
                --csvfile ./data/refined_set.csv --model kdeep \
                --gpu_ids 0,1,2,3,4,5,6,7 --batch_size 256 --nThreads 56 \
                --lr 0.0001 --niter 50 --niter_decay 5 \
                --channels kdeep --rvdw 2 \
                --save_epoch_freq 5 --init_type kaiming
