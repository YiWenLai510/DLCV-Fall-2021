import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result",  type=str)
    parser.add_argument("name",  type=str)
    args = parser.parse_args()

    with open(args.result, "w") as f:
        f.write("id,label,coords\n")
        patient_id = sorted(os.listdir("./skull/test/"))
        for pid in patient_id:
            fracture = False
            dir = os.path.join("./skull/test/", pid)
            id = sorted(os.listdir(dir))
            for i in id:
                labels_path = os.path.join(f"./yolov5/runs/detect/{args.name}/labels", f"{i[:-4]}.txt")
                if os.path.isfile(labels_path):
                    fracture = True
                    break
            for i in id:
                coord = ""
                labels_path = os.path.join(f"./yolov5/runs/detect/{args.name}/labels", f"{i[:-4]}.txt")
                if os.path.isfile(labels_path):
                    with open(labels_path, 'r') as l:
                        lines = l.readlines()
                        for line in lines:
                            line = line.split()
                            w = int(float(line[1]) * 512)
                            h = int(float(line[2]) * 512)
                            coord += f"{w} {h} "
                case = 0
                if fracture:
                    case = 1 if len(coord) else -1
                f.write(f"{i[:-4]},{case},{coord}\n")