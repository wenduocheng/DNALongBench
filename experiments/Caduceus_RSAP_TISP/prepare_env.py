import os
import sys
import argparse
import shutil


def check_and_copy(ori_path, dst_path):
    if os.path.isdir(ori_path):
        if not os.path.exists(dst_path):
            print(f"The {dst_path} does not exist. Adding it.")
            os.makedirs(dst_path)
        for file in os.listdir(ori_path):
            check_and_copy(os.path.join(ori_path, file), os.path.join(dst_path, file))
    else:
        if "__pycache__" not in ori_path and "DS_Store" not in ori_path and ".pyc" not in ori_path:
            if os.path.exists(dst_path):
                print(f"The {dst_path} already exists. This file will be overwritten.")
            print(f"Copying {ori_path} to {dst_path}")
            # shutil.copy(ori_path, dst_path)


if __name__ == "__main__":
    caduceus_path = sys.argv[1]

    pretrained_model_path = os.path.join(caduceus_path, "checkpoints", "caduceus-ph_seqlen-131k_d_model-256_n_layer-16")
    if not os.path.exists(pretrained_model_path):
        raise FileNotFoundError(f"The pretrained Caduceus model {pretrained_model_path} does not exist. Please download the model from the official repository and put it in the `checkpoints` directory.")

    pretrained_model_path = os.path.join(caduceus_path, "checkpoints", "caduceus-ps_seqlen-131k_d_model-256_n_layer-16")
    if not os.path.exists(pretrained_model_path):
        raise FileNotFoundError(f"The pretrained Caduceus model {pretrained_model_path} does not exist. Please download the model from the official repository and put it in the `checkpoints` directory.")


    for path in ["configs", "src", "analysis"]:
        ori_path = os.path.join('.', path)
        dst_path = os.path.join(caduceus_path, path)
        check_and_copy(ori_path, dst_path)

    check_and_copy(os.path.join('.', "train.py"), os.path.join(caduceus_path, "train.py"))
    
    print(f"The environment is ready in {caduceus_path}.")