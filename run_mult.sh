
nohup python3.8 -u runner10_MSL.py  --gpu_id 0 --test_train_step2 10  &>  Runner10_0_0.log &
sleep 1s
nohup python3.8 -u runner10_MSL.py  --gpu_id 0 --test_train_step2 10  &>  Runner10_0_1.log &
sleep 1s

nohup python3.8 -u runner10_MSL.py  --gpu_id 1 --test_train_step2 10  &>  Runner10_1_0.log &
sleep 1s
nohup python3.8 -u runner10_MSL.py  --gpu_id 1 --test_train_step2 10  &>  Runner10_1_1.log &
sleep 1s

nohup python3.8 -u runner10.py  --gpu_id 2 --test_train_step2 10 &>  Runner10_2_0.log &
sleep 1s
nohup python3.8 -u runner10.py  --gpu_id 2 --test_train_step2 10 &>  Runner10_2_1.log &
sleep 1s

nohup python3.8 -u runner10.py  --gpu_id 3 --test_train_step2 10 &>  Runner10_3_0.log &
sleep 1s
nohup python3.8 -u runner10.py  --gpu_id 3 --test_train_step2 10 &>  Runner10_3_1.log &
sleep 1s

