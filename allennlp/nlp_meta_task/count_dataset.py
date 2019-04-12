import os
import argparse
import torch

LAYER_NAMES = ['correct_ll_start_outputs.torch', 'correct_start_ll_start_outputs.torch', 'correct_end_ll_start_outputs.torch', 'incorrect_ll_start_outputs.torch']


def count_dataset(input_dir):
    counts = []
    
    for category in LAYER_NAMES:
        file_path = os.path.join(input_dir, category)
        dataset = torch.load(file_path)

        counts.append(len(dataset))
    
    return counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs_dir', help='Relative path to input datasets to count')
    args = parser.parse_args()

    counts = count_dataset(args.inputs_dir)

    print('Total: {}, Correct: {}, Start Correct: {}, End Correct: {}, Incorrect: {}\n'.format(
            sum(counts), counts[0], counts[1], counts[2], counts[3]))