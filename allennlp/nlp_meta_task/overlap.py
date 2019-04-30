import argparse
import os
from bins_viz import create_meta_labels

def jaccard(input1_filepath, input2_filepath):
    # Load and create meta labels from its outputs
    input1 = create_meta_labels(args.input1)
    input2 = create_meta_labels(args.input2)

    # Compute Jaccard similarity measure
    intersect = 0 

    for i in range(len(input1)):
        if input1[i] == input2[i]:
            intersect += 1
    
    jaccard_measure = intersect / (len(input1) + len(input2) - intersect)

    return jaccard_measure
        

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input1_filepath", help="Relative path to first meta prediction input")
    parser.add_argument("--input2_filepath", help="Relative path to first meta prediction input")
    args = parser.parse_args()
    
    print(jaccard(args.input1_filepath, args.input2_filepath))