# 'hw1_data/p1_data/val_50'
# sample_submission.csv
model="hw1_2_best.pt"
wget https://www.dropbox.com/s/x0nu5wrxwepsxrc/hw1_2_best.pt?dl=1 -O hw1_2_best.pt
python3 p2_test.py --test_dir $1  --out_dir $2 --model_path $model

# python3 p1_test.py --img_dir $1  --outfile $2 --model_path model_p
# bash hw1_2.sh hw1_2_inputs hw1_2_test 
# https://www.dropbox.com/s/x0nu5wrxwepsxrc/hw1_2_best.pt?dl=1