# TODO: create shell script for running your ViT testing code
# rm -rf p1_models/*
# Example
# seed 10293
wget https://www.dropbox.com/s/144b2dtqu4vlr2n/hw3_p1_best.pth?dl=1 -O hw3_p1.pt
model='hw3_p1.pt'
python3 p1_test.py --img_folder $1 --output_file $2 --model_path $model
# bash hw3_1.sh hw3_data/p1_data/val p1_test.csv 
# python3 p1_test.py --id 1 --seed 12093
# python3 p1_test.py --id 1 --seed 3
# python3 p1_test.py --id 1 --seed 23
# python3 p1_test.py --id 1 --seed 693
# python3 p1_test.py --id 1 --seed 4372
# python3 p1_test.py --id 1 --seed 200000
# python3 p1_test.py --id 1 --seed 12093
# python3 p1_test.py --id 1 --seed 12093
# python3 p1_train.py  --seed 3 --smoothLoss 0
# python3 p1_train.py  --seed 3 --smoothLoss 1
# python3 p1_train.py  --seed 23 --smoothLoss 0
# python3 p1_train.py  --seed 23 --smoothLoss 1
# python3 p1_train.py  --seed 693 --smoothLoss 0
# python3 p1_train.py  --seed 693 --smoothLoss 1
# python3 p1_train.py  --seed 4372 --smoothLoss 0
# python3 p1_train.py  --seed 4372 --smoothLoss 1
# python3 p1_train.py  --seed 200000 --smoothLoss 0
# python3 p1_train.py  --seed 148594 --smoothLoss 0
# python3 p1_train.py  --seed 9394 --smoothLoss 0
# python3 p1_train.py  --seed 20234 --smoothLoss 0
# python3 p1_train.py  --seed 904932 --smoothLoss 0
# python3 p1_train.py  --seed 200 --smoothLoss 0
# python3 p1_train.py  --seed 9 --smoothLoss 0
# python3 p1_train.py  --seed 73 --smoothLoss 0
# python3 p1_train.py  --seed 200 --smoothLoss 0
# python3 p1_train.py  --seed 200000 --smoothLoss 1
# python3 p1_train.py  --seed 12093 --smoothLoss 0
# python3 p1_train.py  --seed 12093 --smoothLoss 1
# python3 p1_test.py --id 2
# python3 p1_test.py --id 3
# python3 p1_test.py --id 4
# python3 p1_test.py --id 5
# python3 p1_test.py --id 6
