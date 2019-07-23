import torch
import argparse
import os
import textacy
import pandas as pd
from textacy.text_stats import TextStats
from bins_viz import create_meta_labels
import IPython


# Global Parameters
CORRECT_META_FILE = 'meta_correct_outputs.torch'
INCORRECT_META_FILE = 'meta_incorrect_outputs.torch'
LAYER_NAME = 'inputs.torch'
CORRECT = 'correct_'
INCORRECT = 'incorrect_'


def p_complexity(args):
    correct_meta_labels = create_meta_labels(os.path.join(args.meta_outputs_dir, CORRECT_META_FILE))
    incorrect_meta_labels = create_meta_labels(os.path.join(args.meta_outputs_dir, INCORRECT_META_FILE))

    correct_outputs = torch.load(os.path.join(args.test_outputs_dir, CORRECT + LAYER_NAME), map_location='cpu')
    incorrect_outputs = torch.load(os.path.join(args.test_outputs_dir, INCORRECT + LAYER_NAME), map_location='cpu')

    correct_cmplx = []
    incorrect_cmplx = []

    for output in correct_outputs:
        psg = output['metadata'][0]['original_passage']
        doc = textacy.make_spacy_doc(psg)
        ts = TextStats(doc)
        cur_cmplx = ts.readability_stats['flesch_kincaid_grade_level']
        correct_cmplx.append(cur_cmplx)
    
    for output in incorrect_outputs:
        psg = output['metadata'][0]['original_passage']
        doc = textacy.make_spacy_doc(psg)
        ts = TextStats(doc)
        cur_cmplx = ts.readability_stats['flesch_kincaid_grade_level']
        incorrect_cmplx.append(cur_cmplx)
    
    correct_cmplx_dict = {'Complexity': correct_cmplx, 'Meta Prediction': correct_meta_labels, 'Base Network Prediction': [1] * len(correct_cmplx)}
    incorrect_cmplx_dict = {'Complexity': incorrect_cmplx, 'Meta Prediction': incorrect_meta_labels, 'Base Network Prediction': [0] * len(incorrect_cmplx)}

    correct_cmplx_df = pd.DataFrame.from_dict(correct_cmplx_dict)
    incorrect_cmplx_df = pd.DataFrame.from_dict(incorrect_cmplx_dict)

    return correct_cmplx_df, incorrect_cmplx_df


def q_complexity(args):
    correct_meta_labels = create_meta_labels(os.path.join(args.meta_outputs_dir, CORRECT_META_FILE))
    incorrect_meta_labels = create_meta_labels(os.path.join(args.meta_outputs_dir, INCORRECT_META_FILE))

    correct_outputs = torch.load(os.path.join(args.test_outputs_dir, CORRECT + LAYER_NAME), map_location='cpu')
    incorrect_outputs = torch.load(os.path.join(args.test_outputs_dir, INCORRECT + LAYER_NAME), map_location='cpu')

    correct_cmplx = []
    incorrect_cmplx = []

    for output in correct_outputs:
        q = ' '.join(output['metadata'][0]['question_tokens'])
        doc = textacy.make_spacy_doc(q)
        ts = TextStats(doc)
        cur_cmplx = ts.readability_stats['flesch_kincaid_grade_level']
        correct_cmplx.append(cur_cmplx)
    
    for output in incorrect_outputs:
        q = ' '.join(output['metadata'][0]['question_tokens'])
        doc = textacy.make_spacy_doc(q)
        ts = TextStats(doc)
        cur_cmplx = ts.readability_stats['flesch_kincaid_grade_level']
        incorrect_cmplx.append(cur_cmplx)
    
    correct_cmplx_dict = {'Complexity': correct_cmplx, 'Meta Prediction': correct_meta_labels, 'Base Network Prediction': [1] * len(correct_cmplx)}
    incorrect_cmplx_dict = {'Complexity': incorrect_cmplx, 'Meta Prediction': incorrect_meta_labels, 'Base Network Prediction': [0] * len(incorrect_cmplx)}

    correct_cmplx_df = pd.DataFrame.from_dict(correct_cmplx_dict)
    incorrect_cmplx_df = pd.DataFrame.from_dict(incorrect_cmplx_dict)

    return correct_cmplx_df, incorrect_cmplx_df

def p_q_complexity(args):
    p_correct_cmplx_df, p_incorrect_cmplx_df = p_complexity(args)
    q_correct_cmplx_df, q_incorrect_cmplx_df = q_complexity(args)

    # Sum complexity of passage and question into 1 df
    p_correct_cmplx_df['Complexity'] += q_correct_cmplx_df['Complexity']
    p_incorrect_cmplx_df['Complexity'] += q_incorrect_cmplx_df['Complexity']

    return p_correct_cmplx_df, p_incorrect_cmplx_df


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_outputs_dir", help="Relative path to meta outputs directory")
    parser.add_argument("--test_outputs_dir", help="Relative path to directory with intermediate/final outputs")
    parser.add_argument("--results_dir", help="Relative path to directory for saving plots")
    parser.add_argument("--class", help="question / passage / both (complexity)")
    args = parser.parse_args()
    
    if args.class == 'passage':
        correct_df, incorrect_df = p_complexity(args)
    elif args.class == 'question':
        correct_df, incorrect_df = q_complexity(args)
    else:
        correct_df, incorrect_df = p_q_complexity(args)
    
    IPython.embed()