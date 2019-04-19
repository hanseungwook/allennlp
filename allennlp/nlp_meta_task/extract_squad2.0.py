#!/usr/bin/python3

import json
import sys
import logging
import numpy as np
import argparse
from split_data import count_qs

# Setting up logger
LOGGER = logging.getLogger(__name__)
out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
out_hdlr.setLevel(logging.INFO)
LOGGER.addHandler(out_hdlr)
LOGGER.setLevel(logging.INFO)

# Extract all impossible questions/contexts from Squad 2.0 dataset
def extract_impos(file_path, output_file):
    with open(file_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']

        LOGGER.info("Reading and extracting impossible items from the dataset")

        # Creating new json object
        dataset_impos = { "data" : [] }

        for article in dataset:
            paragraph_list = article['paragraphs']
            impos_pars = []

            for paragraph in paragraph_list:
                impos_qs = []
                impos_idx = []
                question_idx = 0

                # Find all question indices at which is_impossible == True
                for question in paragraph['qas']:
                    if question['is_impossible'] == True:
                        impos_idx.append(question_idx)

                    question_idx += 1
                
                if impos_idx:
                    print(len(paragraph['qas']))
                    print(impos_idx)
                    
                    impos_qs = [paragraph['qas'][i] for i in impos_idx]
                    impos_par_dict = {'qas': impos_qs, 'context': paragraph['context']}
                    impos_pars.append(impos_par_dict)
            
            if impos_pars:
                impos_article_dict = {'title': article['title'], 'paragraphs': impos_pars}
                dataset_impos['data'].append(impos_article_dict)

        # Save train and val datasets
        with open(output_file, 'w') as impos_file:
            LOGGER.info("Saving impossible dataset")
            json.dump(dataset_impos, impos_file)


        return dataset_impos

# Extract all impossible questions/contexts from Squad 2.0 dataset
def extract_pos(file_path, output_file):
    with open(file_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']

        LOGGER.info("Reading and extracting impossible items from the dataset")

        # Creating new json object
        dataset_pos = { "data" : [] }

        for article in dataset:
            paragraph_list = article['paragraphs']
            pos_pars = []

            for paragraph in paragraph_list:
                pos_qs = []
                pos_idx = []
                question_idx = 0

                # Find all question indices at which is_impossible == True
                for question in paragraph['qas']:
                    if question['is_impossible'] == False:
                        pos_idx.append(question_idx)

                    question_idx += 1
                
                if pos_idx:
                    print(len(paragraph['qas']))
                    print(pos_idx)
                    
                    pos_qs = [paragraph['qas'][i] for i in pos_idx]
                    pos_par_dict = {'qas': pos_qs, 'context': paragraph['context']}
                    pos_pars.append(pos_par_dict)
            
            if pos_pars:
                pos_article_dict = {'title': article['title'], 'paragraphs': pos_pars}
                dataset_pos['data'].append(pos_article_dict)

        # Save train and val datasets
        with open(output_file, 'w') as pos_file:
            LOGGER.info("Saving possible dataset")
            json.dump(dataset_pos, pos_file)


        return dataset_pos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_filepath", help="Path to dataset to split")
    parser.add_argument("--output_file", help='Name of output file')
    args = parser.parse_args()

    dataset_impos = extract_pos(args.dataset_filepath, args.output_file)
    print(count_qs(args.output_file))
    