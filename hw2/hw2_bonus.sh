# TODO: create shell script for running your improved UDA model
# rm -rf p3_bonus_save_models/*
# Example
wget https://www.dropbox.com/s/2plgmvxyzxeu58z/hw3_p3_bonus_feat_S_svhn_T_mnistm.pth?dl=1 -O hw2_bonus_mnistm_feature.pt
wget https://www.dropbox.com/s/vk8stuokyti11dj/hw2_p3_bonus_label_S_svhn_T_mnistm.pth?dl=1 -O hw2_bonus_mnistm_label.pt
wget https://www.dropbox.com/s/q3evde2ymrv1hqf/hw2_p3_bonus_feat_S_mnistm_T_usps.pth?dl=1 -O hw2_bonus_usps_feature.pt
wget https://www.dropbox.com/s/zy9yxebqoj0xv9b/hw2_p3_bonus_label_S_mnistm_T_usps.pth?dl=1 -O hw2_bonus_usps_label.pt

python3 p3_bonus_test.py --image_folder $1 --target_domain $2 --output_file $3
# python3 p3_test.py  --image_folder $1 --target_domain $2 --output_file $3
# bash  hw2_bonus.sh hw2_data/digits/svhn/test svhn svhn_test_b.csv
# bash  hw2_bonus.sh hw2_data/digits/mnistm/test mnistm mnistm_test_b.csv
# bash  hw2_bonus.sh hw2_data/digits/usps/test usps usps_test_b.csv