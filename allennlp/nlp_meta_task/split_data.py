import json
import logging
import numpy as np
import argparse

logger = logging.getLogger(__name__)

def count_paragraphs(file_path):
    with open(file_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']

        logger.info("Reading and counting examples the dataset")

        total_num_paragraphs = 0

        for article in dataset:
            cur_num_paragraphs = len(article['paragraphs'])
            total_num_paragraphs += cur_num_paragraphs

        return total_num_paragraphs

def count_qs(file_path):
    with open(file_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']

        logger.info("Reading and counting examples the dataset")

        total_num_qs = 0

        for article in dataset:
            for paragraph_json in article['paragraphs']:    
                cur_num_qs = len(paragraph_json['qas'])
                total_num_qs += cur_num_qs

        return total_num_qs


def split_data_by_par(file_path, split_ratio):
    with open(file_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']

        logger.info("Reading and splitting the dataset")

        # Creating new json objects
        dataset_train = { "data" : [] }
        dataset_val = { "data" : [] }

        # Generating random numbers for extraction
        num_par = count_paragraphs(file_path)
        num_par_val = int(np.floor(num_par * split_ratio))
        
        val_par_indices = np.random.choice(num_par, num_par_val, replace=False)

        cur_par_start = 0
        for article in dataset:
            paragraph_list = article['paragraphs']
            num_par = len(paragraph_list)
            cur_par_end = cur_par_start + num_par

            # Get paragraph indices that fall into this article
            sel_par_indices = val_par_indices[(val_par_indices >= cur_par_start) & (val_par_indices < cur_par_end)]
            norm_sel_par_indices = sel_par_indices - cur_par_start

            # Extracting respective paragraphs
            train_par = list(np.delete(paragraph_list, norm_sel_par_indices))
            val_par = list(np.take(paragraph_list, norm_sel_par_indices))
            
            train_dict = {}
            val_dict = {}
            cur_title = article["title"]

            # Inserting respective paragraphs in given structure
            if len(train_par) > 0 and len(val_par) > 0:    
                train_dict["title"] = cur_title
                val_dict["title"] = cur_title

                train_dict["paragraphs"] = train_par
                val_dict["paragraphs"] = val_par

                dataset_train["data"].append(train_dict)
                dataset_val["data"].append(val_dict)
            
            elif len(train_par) == 0:
                val_dict["title"] = cur_title
                val_dict["paragraphs"] = val_par
                dataset_val["data"].append(val_dict)
            
            else:
                train_dict["title"] = cur_title
                train_dict["paragraphs"] = train_par
                dataset_train["data"].append(train_dict)

            # Update paragraph start index
            cur_par_start = cur_par_end

        # Save train and val datasets
        with open('dataset_train.json', 'w') as split_train:
            logger.info("Saving train dataset")
            json.dump(dataset_train, split_train)
        
        with open('dataset_val.json', 'w') as split_val:
            logger.info("Saving validation dataset")
            json.dump(dataset_val, split_val)

        return dataset_train, dataset_val

# TODO: Split data by using # of questions
def split_data_by_q(file_path, split_ratio):
    with open(file_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']

        logger.info("Reading and splitting the dataset")

        # Creating new json objects
        dataset_val = { "data" : [] }
         
        total_q = count_qs(file_path)
        val_q = int(np.floor(total_q * split_ratio))
        cur_q = 0

        while cur_q < val_q:
            val_dict = {}

            rand_title_ind = np.random.randint(len(dataset), size=1)[0]

            article = dataset[rand_title_ind]
            title = article['title']
            paragraphs_list = article['paragraphs']
            
            rand_par_ind = np.random.randint(len(paragraphs_list), size=1)[0]
            
            cur_q += len(paragraphs_list[rand_par_ind]['qas'])
            val_dict['title'] = title

            if (len(paragraphs_list) <= 1):
                dataset.pop(rand_title_ind)
    
                val_dict['paragraphs'] = paragraphs_list
            else:
                paragraph = paragraphs_list.pop(rand_par_ind)
                
                val_dict['paragraphs'] = [paragraph]
            
            dataset_val['data'].append(val_dict)

        # Save train and val datasets
        with open('dataset_train_q.json', 'w') as split_train:
            logger.info("Saving train dataset")
            dataset_train = {'data' : dataset}
            json.dump(dataset_train, split_train)
        
        with open('dataset_val_q.json', 'w') as split_val:
            logger.info("Saving validation dataset")
            json.dump(dataset_val, split_val)

        return dataset, dataset_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_filepath", help='Path to dataset to split')
    args = parser.parse_args()

    # num_paragraphs = count_paragraphs(args.dataset_filepath)
    # print(num_paragraphs)
    # dataset_train, dataset_val = split_data_by_par(args.dataset_filepath, 0.2)
    # num_qs = count_qs(args.dataset_filepath)
    # print(num_qs)

    # Split data by questions
    # dataset_train, dataset_val = split_data_by_q(args.dataset_filepath, 0.2)

    dataset_train, dataset_val = split_data_by_q(args.dataset_filepath, 0.5)
    print(count_qs('./dataset_train_q.json'))
    print(count_qs('./dataset_val_q.json'))
    