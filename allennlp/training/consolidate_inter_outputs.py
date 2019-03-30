import torch
import os
import argparse

def consolidate_batches(file_dir):
    for inter_layer in next(os.walk(file_dir))[1]:
        print(inter_layer)
        consolidated_tensors = []

        inter_dir = os.path.join(os.getcwd(), inter_layer)

        files = os.listdir(inter_dir)
        files.sort()

        for f in files:
            t = torch.load(inter_dir + '/' + f)
            for x in t:
                consolidated_tensors.append(x)

        torch.save(consolidated_tensors, inter_dir + '/' + inter_layer + '.torch')
        consolidated_tensors.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Consolidate intermediate output batches')
    parser.add_argument('--file_dir', help='Folder with intermediate output directories to consolidate')
    
    args = parser.parse_args()
    consolidate_batches(args.file_dir)