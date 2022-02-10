import csv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--acc_csv_path",  type=str, required=True)
parser.add_argument("--coords_csv_path",  type=str, required=True)
parser.add_argument("--output_csv_path", type=str, required=True)
args = parser.parse_args()
acc_csv_path = args.acc_csv_path
coords_csv_path = args.coords_csv_path
output_csv_path = args.output_csv_path

IDs = []
labels = []
coords = []
with open(acc_csv_path) as acc_file:
    acc = csv.reader(acc_file, delimiter=",")
    for row in acc:
        if acc.line_num == 1:
            continue
        image_id, image_label, image_coords = row
        IDs.append(image_id)
        labels.append(image_label)

cnt = 0.
case = [0.,0.,0.,0.,0.,0.]
with open(coords_csv_path) as f1_file:
    f1 = csv.reader(f1_file, delimiter=",")
    for row in f1:
        if f1.line_num == 1:
            continue

        cnt += 1.
        image_id, image_label, image_coords = row
        idx = f1.line_num-2
        assert IDs[idx] == image_id
        if(labels[idx] == "-1"):
            if(len(image_coords)):
                labels[idx] = 1
                coords.append(image_coords)
                case[0] += 1
            else:
                labels[idx] = -1
                coords.append("")
                case[1] += 1
        elif(labels[idx] == "0"):
            if(len(image_coords)):
                labels[idx] = 0
                coords.append("")
                case[2] += 1
            else:
                labels[idx] = 0
                coords.append("")
                case[3] += 1
        else:
            if(len(image_coords)):
                labels[idx] = 1
                coords.append(image_coords)
                case[4] += 1
            else:
                labels[idx] = -1
                coords.append("")
                case[5] += 1
case = np.array(case)
case /= cnt
# print(f'case1: {case[0]}, case2: {case[1]}, case3: {case[2]}, case4: {case[3]}, case5: {case[4]}, case6: {case[5]}')

with open(output_csv_path, "w") as f:
    f.write("id,label,coords\n")
    for i, l, c in zip(IDs, labels, coords):
        f.write(f"{i},{l},{c}\n")