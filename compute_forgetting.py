import os
import copy
import sys
import argparse
import shutil
import time
import math
import random
import numpy as np
import torch


parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--result_folder', type=str, default='/datasets/work/d61-eif/source/', help='path to custom dataset')
parser.add_argument('--n_task', type=int, default=10, help='dataset name')
parser.add_argument('--n_class_per_task', type=int, default=10, help='n_class_per_task')

opt = parser.parse_args()

results = []

for task in range(opt.n_task):
    with open(os.path.join(opt.result_folder, "acc_buffer_{}.txt".format(task)), "r") as result_file:
        lines = result_file.readlines()
        results_task = [0]*opt.n_class_per_task*opt.n_task
        i = 0
        for l in lines:
            as_list = l.split(" ")
            if len(as_list) == 1: 
                # print(as_list)
                results_task[i] = float(as_list[0].replace('\n', ''))
                i = i + 1
        results.append(results_task)
                

print(results)

def forgetting(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)


print(forgetting(results))

