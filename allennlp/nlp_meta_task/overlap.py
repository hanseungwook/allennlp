import argparse
import os
import numpy as np
from bins_viz import create_meta_labels

def jaccard(input1_filepath, input2_filepath):
    # Load and create meta labels from its outputs
    input1 = create_meta_labels(input1_filepath)
    input2 = create_meta_labels(input2_filepath)

    # Compute Jaccard similarity measure
    intersect = 0 

    for i in range(len(input1)):
        if input1[i] == input2[i]:
            intersect += 1
    
    jaccard_measure = intersect / (len(input1) + len(input2) - intersect)

    return jaccard_measure

def jaccard_meta_correct(input1_filepath, input2_filepath, data_class=None):
    expected = None
    if data_class == 'correct':
        expected = 1
    elif data_class == 'incorrect':
        expected = 0
    else:
        raise Exception('No data class specified for overlap')

    # Load and create meta labels from its outputs
    input1 = np.array(create_meta_labels(input1_filepath))
    input2 = np.array(create_meta_labels(input2_filepath))

    input1_correct = input1[input1 == expected]
    input2_correct = input2[input2 == expected]

    # Compute Jaccard similarity measure
    intersect = 0 

    for i in range(len(input1)):
        if input1[i] == input2[i]:
            intersect += 1
    
    jaccard_measure = intersect / (len(input1_correct) + len(input2_correct) - intersect)

    return jaccard_measure

# TODO: meta correct for LSTM intersection with meta correct for last layer / min(of the two)

def overlap_of_smaller(input1_filepath, input2_filepath):
    # Load and create meta labels from its outputs
    input1 = create_meta_labels(input1_filepath)
    input2 = create_meta_labels(input2_filepath)

    # Compute similarity measure
    intersect = 0 

    for i in range(len(input1)):
        if input1[i] == input2[i]:
            intersect += 1
    
    measure = intersect / min(len(input1), len(input2))

    return measure

def overlap_meta_correct_smaller(input1_filepath, input2_filepath, data_class=None):
    expected = None
    if data_class == 'correct':
        expected = 1
    elif data_class == 'incorrect':
        expected = 0
    else:
        raise Exception('No data class specified for overlap')

    # Load and create meta labels from its outputs
    input1 = np.array(create_meta_labels(input1_filepath))
    input2 = np.array(create_meta_labels(input2_filepath))

    input1_correct = input1[input1 == expected]
    input2_correct = input2[input2 == expected]

    # Compute similarity measure
    intersect = 0 

    for i in range(len(input1)):
        if input1[i] == input2[i] and input1[i] == expected:
            intersect += 1
    
    measure = intersect / min(len(input1_correct), len(input2_correct))

    return measure

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input1_filepath", help="Relative path to first meta prediction input")
    parser.add_argument("--input2_filepath", help="Relative path to first meta prediction input")
    parser.add_argument("--data_class", help="correct or incorrect data class")
    args = parser.parse_args()
    
    print('Jaccard: {}'.format(jaccard_meta_correct(args.input1_filepath, args.input2_filepath, args.data_class)))
    print('Overlap of smaller: {}'.format(overlap_meta_correct_smaller(args.input1_filepath, args.input2_filepath, args.data_class)))