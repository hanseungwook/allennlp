import os
import torch
import IPython
import argparse
from scipy.stats import entropy
from itertools import compress
from itertools import cycle
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt


# Global Parameters
CORRECT_META_FILE = 'meta_correct_outputs.torch'
INCORRECT_META_FILE = 'meta_incorrect_outputs.torch'
#LAYER_NAMES = ['ll_start_outputs.torch', 'll_end_outputs.torch']
LAYER_NAME = 'outputs.torch'
CORRECT = 'correct_'
INCORRECT = 'incorrect_'
BIN_NAMES = ['b_correct_m_correct', 'b_correct_m_incorrect', 'b_incorrect_m_correct', 'b_incorrect_m_incorrect']
FIG_IDX = 0
COLORS = cycle(['b', 'r', 'g', 'y'])


def create_meta_labels(output_filepath):
    outputs = outputs = torch.load(output_filepath, map_location='cpu')

    meta_labels = torch.tensor([])
    for batch_output in outputs:
        batch_output_size = batch_output.shape[0]
        batch_output_labels = batch_output.max(1, keepdim=True)[1].reshape(batch_output_size)
        
        if meta_labels.shape[0] == 0:
            meta_labels = batch_output_labels
        
        else:
            meta_labels = torch.cat((meta_labels, batch_output_labels))
    
    return meta_labels


def construct_4_bins(test_outputs_dir, side, correct_meta_labels, incorrect_meta_labels,):
    correct_outputs = torch.load(os.path.join(test_outputs_dir, CORRECT + LAYER_NAME), map_location='cpu')
    incorrect_outputs = torch.load(os.path.join(test_outputs_dir, INCORRECT + LAYER_NAME), map_location='cpu')

    if side == 'start':
        LAYER = 'span_start_probs'
    elif side == 'end':
        LAYER = 'span_end_probs'

    cor_cor = preprocess_outputs([x[LAYER] for x in list(compress(correct_outputs, correct_meta_labels))])
    cor_incor = preprocess_outputs([x[LAYER] for x in list(compress(correct_outputs, [not i for i in correct_meta_labels]))])
    incor_cor = preprocess_outputs([x[LAYER] for x in list(compress(incorrect_outputs, [not i for i in incorrect_meta_labels]))])
    incor_incor = preprocess_outputs([x[LAYER] for x in list(compress(incorrect_outputs, incorrect_meta_labels))])

    bins = [cor_cor, cor_incor, incor_cor, incor_incor]

    return bins


def preprocess_outputs(outputs):
    processed_outputs = []

    for i in outputs:
        processed_data = i.view(i.shape[1])
        processed_outputs.append(processed_data)
    
    return processed_outputs


def create_viz(results_dir, y, data_name):
    global FIG_IDX
    plt.figure(FIG_IDX)
    plt.subplot(1,1,1)
    plt.ylabel(data_name)

    for i in range(len(y)):
        plt.subplot(1, 4, i+1)
        plt.scatter(list(range(len(y[i]))), y[i], color=next(COLORS), s=1)
        plt.xlabel(BIN_NAMES[i])
        
    plt.savefig(os.path.join(results_dir, data_name + '_' + BIN_NAMES[i] + '.png'))
    FIG_IDX += 1


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_outputs_dir", help="Relative path to model weights file")
    parser.add_argument("--test_outputs_dir", help="Relative path to directory with intermediate/final outputs")
    parser.add_argument("--results_dir", help="Relative path to directory for saving plots")
    args = parser.parse_args()

    correct_meta_labels = create_meta_labels(os.path.join(args.meta_outputs_dir, CORRECT_META_FILE))
    incorrect_meta_labels = create_meta_labels(os.path.join(args.meta_outputs_dir, INCORRECT_META_FILE))

    start_bins = construct_4_bins(args.test_outputs_dir, 'start', correct_meta_labels, incorrect_meta_labels)
    end_bins = construct_4_bins(args.test_outputs_dir, 'end', correct_meta_labels, incorrect_meta_labels)

    means = [[] for i in range(4)]
    stds = [[] for i in range(4)]
    entropies = [[] for i in range(4)]
    maxes = [[] for i in range(4)]

    for x in range(4):
        for i in start_bins[x]:
            means[x].append(i.mean().numpy())
            stds[x].append(i.std().numpy())
            maxes[x].append(i.max().numpy())
            entropies[x].append(entropy(i.numpy()))

    try:
        os.mkdir(args.results_dir)
    except:
        raise Exception('Could not create results directory')

    create_viz(args.results_dir, means, 'means')
    create_viz(args.results_dir, stds, 'standard deviations')
    create_viz(args.results_dir, maxes, 'maximum probabilities')
    create_viz(args.results_dir, entropies, 'entropies')