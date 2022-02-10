# DLCV Final Project ( Skull Fractracture Detection )

# How to train a transformer encoder for labeling?
First, install the requirements

    $ pip install -r requirements1.txt
Second, train a classifier with feature extractor and transformer encoder

    $ python3 train_transformer.py --model_path <your model path>
Last, test the final model.
Also, we provide a pretrained model with acc 0.854% called transformer.ckpt

    $ python3 test_transformer.py --model_path <your model path> --output_csv_path <output csv path>

# How to train or test a YOLOv5 model
- The initial path is under `final-project-challenge-1-conbaconcon/` for all the following steps.
- Clone repo [ultralytics/yolov5](https://github.com/ultralytics/yolov5) and install requirement
    ```
    $ git clone https://github.com/ultralytics/yolov5.git
    $ pip install -r yolov5/requirements.txt
    ```
- Prepare directories for yolo dataset
    ```
    $ mkdir <data_dir> <data_dir>/images <data_dir>/images/train <data_dir>/images/val <data_dir>/images/test <data_dir>/labels <data_dir>/labels/train <data_dir>/labels/val
    ```
- Move the data to their respective folders and ramdomly split the training data by 90%/10%, where 10% of training data are for validation
    ```
    $ python3 yolo_dataset.py <data_dir>
    ```
- Create a new file called `<data_name>.yaml` under `yolov5/data`. The contents of `<data_name>.yaml` are shown as follow
    ```
    # <data_dir> is the same as above
    train: ../<data_dir>/images/train/ 
    val:  ../<data_dir>/images/val/
    test: ../<data_dir>/images/test/

    # number of classes
    nc: 1

    # class names
    names: ["fracture"]
    ```
- Modify network architecture by changing variable `nc` in `yolov5/models/yolov5m.yaml`
    ```
    nc: 1
    ```
- Train from scratch and test with the provided skull data
    ```
    $ cd yolov5
    $ python3 train.py --img 512 --cfg yolov5m.yaml --hyp hyp.scratch.yaml --batch <batch_size> --epochs <#_of_epoch> --data <data_name>.yaml --workers <#_of_worker> --name <train_name>
    $ python3 detect.py --source ../<data_dir>/images/test/ --weights runs/train/<train_name>/weights/best.pt --conf <level_of_confidence> --name <result_name> --save-txt

    ```
    F.Y.I. We set `<batch_size>` to 16, `<#_of_epoch>` to 1200, and `<#_of_worker>` to 8 during training and set `<level_of_confidence>` to 0.6 during testing.
- We have provided you our weight, `best.pt`, with best performance
    ```
    $ cd yolov5
    $ python3 detect.py --source ../<data_dir>/images/test/ --weights ../best.pt --conf <level_of_confidence> --name <result_name> --save-txt
    ```
- Change the detection result of yolo to required format
    ```
    $ python3 yolo_to_csv.py <coords_csv>.csv <result_name>
    ```

# How to merge .csv files from above
    $ python3 merge.py --acc_csv_path <acc_csv_path> --coords_csv_path <coords_csv_path> --output_csv_path <output_csv_path>

# Experiment Results
|  Method | Case-level Acc | F1 score |
| -------- | -------- | -------- |
| Transformer + YOLOv5m (Merge) (Ours) | **0.854** | **0.67** |
| YOLOv5m | 0.59 | 0.649 |
| YOLOv5s | 0.5 | 0.55 |
| Resnet34 + 2 layer BiLSTM | 0.64 | - |
|BYOL + resnet50 pretrain & finetune|0.75| - |
|3D ResNet18 | 0.70 | 0.40 |
|2D ResNet18 | 0.76 | 0.35 |


# Usage
To start working on this final project, you should clone this repository into your local machine by using the following command:

    git clone https://github.com/DLCV-Fall-2021/final-project-challenge-1-conbaconcon.git


For more details, please click [this link] to view the slides of Final Project - Skull Fracture Detection. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `skull`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link] and unzip the compressed file manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> 1. Please do not disclose the dataset! Also, do not upload your get_dataset.sh to your (public) Github.
> 2. You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `skull` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

> 🆕 ***NOTE***  
> For the sake of conformity, please use the `python3` command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.

### Evaluation Code
In the starter code of this repository, we have provided a python script for evaluating the results for this project. For Linux users, use the following command to evaluate the results.
```bash
python3 for_students_eval.py --pred_file <path to your prediction csv file> --gt_file <path to the ground-truth csv file>
```

# Submission Rules
### Deadline
110/1/18 (Tue.) 23:59 (GMT+8)
    
# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under Final Project FAQ section in NTU Cool Discussion
