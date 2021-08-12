import os
import glob
import argparse
import shutil

from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, help="path to image folder")

    args = parser.parse_args()

    return args


def main(img_dir):
    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    img_files = sorted(img_files)

    train_size = int((len(img_files)+1)*.80)

    train_files = img_files[:train_size] #Remaining 80% to training set
    test_files = img_files[train_size:] #Splits 20% data to test set

    train_dir = os.path.join(img_dir, "training", "frames")
    test_dir = os.path.join(img_dir, "testing", "frames")

    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    for train_file in train_files:
        shutil.move(train_file, os.path.join(train_dir, os.path.basename(train_file)))

    for test_file in test_files:
        shutil.move(test_file, os.path.join(test_dir, os.path.basename(test_file)))


if __name__ == "__main__":
    args = parse_args()
    main(args.img_dir)
