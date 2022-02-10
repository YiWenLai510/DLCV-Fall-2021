# 'hw1_data/p1_data/val_50'
# sample_submission.csv
model="hw1_1_best.pt"
wget  https://www.dropbox.com/s/vi1x6gklf9e82pa/hw1_1_best.pt?dl=1 -O hw1_1_best.pt
python3 p1_test.py --img_dir $1  --outfile $2 --model_path $model

# python3 p1_test.py --img_dir $1  --outfile $2 --model_path model_p
# bash hw1_1.sh hw1_data/p1_data/val_50 sample_submission.csv
# https://www.dropbox.com/s/vi1x6gklf9e82pa/hw1_1_best.pt?dl=1