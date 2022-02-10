# TODO: create shell script for running your DANN model

# Example
# rm -rf p3_adaptive_save_models/*
wget https://www.dropbox.com/s/xlaxbkuphdq4ryq/hw2_p3_feature_S_mnistm_T_usps.pth?dl=1 -O hw2_p3_usps_feature.pt
wget https://www.dropbox.com/s/e29ibuz0484et37/hw2_p3_label_S_mnistm_T_usps.pth?dl=1 -O hw2_p3_usps_label.pt
wget https://www.dropbox.com/s/r616mep6mlcs18e/hw2_p3_feature_S_svhn_T_mnistm.pth?dl=1 -O hw2_p3_mnistm_feature.pt
wget https://www.dropbox.com/s/3hi2qiupqgavedy/hw2_p3_label_S_svhn_T_mnistm.pth?dl=1 -O hw2_p3_mnistm_label.pt
wget https://www.dropbox.com/s/eqea12gg92gmqwk/hw2_p3_feature_S_usps_T_svhn.pth?dl=1 -O hw2_p3_svhn_feature.pt
wget https://www.dropbox.com/s/0q52okigosh9o98/hw2_p3_label_S_usps_T_svhn.pth?dl=1 -O hw2_p3_svhn_label.pt

python3 p3_test.py  --image_folder $1 --target_domain $2 --output_file $3
# bash  hw2_p3.sh hw2_data/digits/svhn/test svhn svhn_test.csv
# bash  hw2_p3.sh hw2_data/digits/mnistm/test mnistm mnistm_test.csv
# bash  hw2_p3.sh hw2_data/digits/usps/test usps usps_test.csv