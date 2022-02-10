# rm -rf p2_result/*
wget https://www.dropbox.com/s/ol2r87v2hr08e98/hw2_p2_best.pth?dl=1 -O hw2_p2.pt
model='hw2_p2.pt'
python3 p2_test.py --result_dir $1 --model_path $model
