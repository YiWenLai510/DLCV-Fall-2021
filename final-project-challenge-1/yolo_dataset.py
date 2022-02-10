import os
import json
import random
from tqdm import tqdm
from PIL import Image
import numpy as np
import imageio
import csv
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data",  type=str)
    args = parser.parse_args()

    with open("./skull/records_train.json", 'r') as json_file:
        data = json.load(json_file)["datainfo"]
        for img_name, value in tqdm(data.items()):
            if value["label"] != 1:
                continue
            t = "train" if random.random() < 0.9 else "val"
            op = os.path.join('./skull/train/', value["path"])
            newp = os.path.join(args.data, "images", t, f"{img_name}.bmp")
            img = np.load(op)
            # img = np.tile(np.expand_dims(img, axis=-1), (1,1,3))
            imageio.imsave(newp, img)
            cs = value["coords"]
            save_file_name = os.path.join(args.data, "labels", t, f"{img_name}.txt")
            with open(save_file_name, 'w') as f:
                for c in cs:
                    f.write(f"0 {c[0]/512} {c[1]/512} 0.1 0.1\n")

    patient_id = sorted(os.listdir("./skull/test/"))
    for pid in patient_id:
        dir = os.path.join("./skull/test/", pid)
        id = sorted(os.listdir(dir))
        for i in id:
            op = os.path.join(dir, i)
            newp = os.path.join(args.data, "images/test", f"{i[:-4]}.bmp")
            img = np.load(op)
            imageio.imsave(newp, img)
