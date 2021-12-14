import os
import os.path as osp


def make_dir(input_dir):
    if not osp.exists(input_dir):
        os.mkdir(input_dir)
