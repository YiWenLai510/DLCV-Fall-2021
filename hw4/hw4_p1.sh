# TODO: create shell script for running your prototypical network
wget  https://www.dropbox.com/s/3ycfhbql3f2ges7/hw4p1_distance_param.pt?dl=1 -O hw4_p1_distance.pt
wget  https://www.dropbox.com/s/rpreehswwupszpd/hw4p1_model_param.pt?dl=1 -O hw4_p1_model.pt
python3 test_testcase.py --test_csv $1 --test_data_dir $2 --testcase_csv $3 --output_csv $4
# bash hw4_p1.sh hw4_data/mini/val.csv hw4_data/mini/val hw4_data/mini/val_testcase.csv p1_final.csv
# Example
# bash hw4_p1.sh $1 $2 $3 $4
# python3 test_testcase.py --N_shot 1
# python3 eval.py p1_1.csv hw4_data/mini/val_testcase_gt.csv
# python3 test_testcase.py --N_shot 5
# python3 eval.py p1_1.csv hw4_data/mini/val_testcase_gt.csv
# python3 test_testcase.py --N_shot 10
# python3 eval.py p1_1.csv hw4_data/mini/val_testcase_gt.csv
# python3 train_fewshot.py --shot 1 --distance eu
# python3 train_fewshot.py --shot 1 --distance cos
# python3 train_fewshot.py --shot 1 --distance param
# python3 train_fewshot.py --shot 5 --distance eu
# python3 train_fewshot.py --shot 5 --distance cos
# python3 train_fewshot.py --shot 5 --distance param
# python3 train_fewshot.py --shot 10 --distance eu
# python3 train_fewshot.py --shot 10 --distance cos
# python3 train_fewshot.py --shot 10 --distance param
# python3 eval.py p1_final.csv hw4_data/mini/val_testcase_gt.csv