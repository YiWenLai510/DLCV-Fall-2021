# rm -rf p1_result/*
wget https://www.dropbox.com/s/iwz17syivf3h3n0/hw2_p1_best.pth?dl=1 -O hw2_p1.pt
model='hw2_p1.pt'
python3 p1_test.py --result_dir $1 --model_path $model
# python3 new_incep.py --folder p1_result
# python3 -m pytorch_fid hw2_data/face/test p1_result 
