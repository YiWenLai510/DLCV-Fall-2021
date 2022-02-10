# TODO: create shell script for running your data hallucination model

# Example
wget https://www.dropbox.com/s/t4zg857ha4o0up6/hw4_p2_model.pth?dl=1 -O hw4_p2_model.pt
# bash hw4_p2.sh hw4_data/office/val.csv hw4_data/office/val p2_result2.csv
python3 p2_test.py --test_csv $1 --test_dir $2 --out_csv $3
# python3 p2_downstream.py --pretrained True --ta_backbone True --classifier_only False
# python3 p2_downstream.py --pretrained True --ta_backbone False --classifier_only False --n_epochs 50
# python3 p2_downstream.py --pretrained True --ta_backbone True --classifier_only True --n_epochs 50
# python3 p2_downstream.py --pretrained True --ta_backbone False --classifier_only True --n_epochs 50
# python3 p2_downstream.py --pretrained False --ta_backbone True --classifier_only False