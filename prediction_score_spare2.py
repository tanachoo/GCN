# Author: yoshi
# Write start: 5/13/2019
# Updated: 7/3/2019

## Script for converting prediction score to table

# Usage: python prediction_score.py --result ./pool_result/target0_20190425/cv_info.gcn.jbl --dataset ./processed_data/target0_20190425/dataset.jbl \
# --node ./processed_data/target0_20190425/dataset_node.csv --cv 0 --scorerank 15000 --train true --output ./pool_result/target0_20190425/score.txt

import joblib
import pandas as pd
import time
import argparse
from scipy import stats
from multiprocessing import Pool, Manager
from functools import partial
import pickle


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict


def build_node_name(filename):
    """ To convert node ID to gene/chemical name """
    print(f'\n== Prep node names list ==\n'
          f'load: {filename}')  # node data='../../processed_data/target0_20190425/dataset_node.csv'
    node_names = []
    with open(filename, 'r') as f:
        for l in f:
            nodes = l.strip()
            node_names.append(nodes)
    print(f'#node_names: {len(node_names)}')
    return node_names


def build_test_label_pairs(filename, cv):
    """ To make test label pair list """
    # import main result data (post caluculation jbl file)
    print(f'\n== Prep test label pairs list ==\n'
          f'load: {filename}\n'
          f'cv fold: {cv}\n')
    result_data = joblib.load(filename)
    test_labels = result_data[cv]['test_labels']
    test_label_pairs = []

    for i in test_labels[0]:
        test_label_pair = (i[0], i[2])
        test_label_pair = tuple(sorted(test_label_pair))
        test_label_pairs.append(test_label_pair)

    print(f'#test_label_pairs: {len(test_label_pairs)}\n'
          f'#original_test_label: {len(test_labels[0])}\n'
          f'(should be same values...)\n'
          f'remove duplicate...\n'
          f'\nremove duplicate...')
    test_label_pairs = list(set(test_label_pairs))  # remove duplicated in list of test_label_pairs
    print(f'#duplicate removed test_label_pairs: {len(test_label_pairs)}\n'
          f'okay... Completed to prep test label list.')
    return test_label_pairs


def build_target_label_pairs(filename):  # args.dataset (input data jbl file)
    """To make all prediction target (train+test) label pair list"""
    # import all edge label data (input data for establish model, train + test) 
    print(f'\n== Prep all target label pairs list ==\n'
          f'load: {filename}')
    input_data = joblib.load(filename)
    label_list = input_data['label_list']
    target_label_pairs = []

    for i in label_list[0]:
        label_pair = (i[0], i[2])
        label_pair = tuple(sorted(label_pair))
        target_label_pairs.append(label_pair)

    print(f'#target_label_pairs: {len(target_label_pairs)}\n'
          f'#original_label_pairs: {len(label_list[0])}\n'
          f'(should be same values...)\n'
          f'\nremove duplicate...')
    target_label_pairs = list(set(target_label_pairs))  # remove duplicated in list of target_label_pairs
    print(f'#duplicated removed target_label_pairs: {len(target_label_pairs)}\n'
          f'okay... Completed to prep target label list.')
    return target_label_pairs


def sort_prediction_score(filename, cv, target_label_pairs, test_label_pairs, scorerank, train):
    """ Sort prediction result array matrix and Set threshold """
    print('\n== Sort predisction score ==')
    print(f'load: {filename}')
    with open(filename, 'rb') as f:  # add for test
        result_data = pickle.load(f)  # add for test
    # result_data = joblib.load(filename)
    print(f'cv fold: {cv}')
    # prediction = result_data[cv]['prediction_data']
    # matrix = prediction[0]
    matrix = result_data  # add for test
    print(f'prediction score matrix shape: {matrix.shape}\n,'
          f'\nprep list of [(score,row,col),(),(),,,,] from prediction score results matrix...')
    dim_row = matrix.shape[0]
    dim_col = matrix.shape[1]
    score_row_col = [(matrix[row, col], row, col) for row in range(dim_row) for col in range(row+1, dim_col)]

    print(f'#scores as adopted: {len(score_row_col)}')  # 480577503
    # print('(should be 480577503...)')
    # Here need to delete CHEBI ID
    row_CHEBI_deleted = [i for i in score_row_col if i[1] < 3071 or i[1] > 14506]
    row_col_CHEBI_deleted = [i for i in row_CHEBI_deleted if i[2] < 3071 or i[2] > 14506]
    print(f'#scores as adopted post removal of CHEBI nodes: {len(row_col_CHEBI_deleted)}')
    
    # sort scores with descending order
    print('\nSort scores and Pick top 2000000...')
    row_col_CHEBI_deleted.sort(reverse=True)  # Sort list based on "score" with a decending order
    score_sort = row_col_CHEBI_deleted[:2000000]  # Cut top list using arbitrary threshold
    # score_sort=sorted(row_col_CHEBI_deleted,reverse=True) # this "sorted" method is slower
    # score_sort=score_sort[:3000000]
    print('okay...done.')

    # Prep target,test,train label list
    train_label_pairs = list(set(target_label_pairs) - set(test_label_pairs))
    
    if train == 'true':
        print('\nTrain labels are included for preparing score-ordred list.\n'
              '#scores including train labels: {len(score_sort)}\n'
              'Cutoff top list by score-rank...\n')
        score_sort_toplist = score_sort[:scorerank]  # args.scorerank: Select top score ranking to export
        print(f'score rank cutoff value: {scorerank}\n'  # should be less than 3,000,000
              f'#score post score-rank cutoff: {len(score_sort_toplist)}\n'
              f'(should be same values...)\n'
              f'Completed to prep prediction score-ordered list including train labels.')
        return score_sort_toplist
    else:
        print('\nTrain labels are excluded for preparing score-ordred list.')
        score_tmp = [i for i in score_sort if (i[1], i[2]) not in train_label_pairs]
        score_tmp.sort(reverse=True)
        score_sort_toplist = score_tmp[:scorerank]
        print(f'#scores post removal of train labels: {len(score_tmp)}\n'
              f'score rank cutoff value: {scorerank}\n'
              f'#src_score_sort_toplist: {len(score_sort_toplist)}\n'
              f'(should be same values...)\n'
              f'Completed to prep prediction score-ordered list w/o train labels.')
        return score_sort_toplist


def convert(score_sort_toplist, target_label_pairs, test_label_pairs, node_names, train, total_list):
    """
    let score-sorted list [(score,row,col),...] convert to table
    total_list = (scores, rows, cols, gene1, gene2, train_edge, test_edge, new_edge)
    """
    print('\n== Start convesion of prediction scores ==')
    train_label_pairs = list(set(target_label_pairs) - set(test_label_pairs))

    if train == 'true':
        print('Train labels are included.')
        for i in score_sort_toplist:
            scores = i[0]
            row = i[1]
            gene1 = node_names[row]
            col = i[2]
            gene2 = node_names[col]
            prediction_label_pair = (row, col)
            if prediction_label_pair in target_label_pairs:
                if prediction_label_pair in test_label_pairs:
                    total_list.append([scores, row, col, gene1, gene2, 0, 1, 0])
                else:
                    total_list.append([scores, row, col, gene1, gene2, 1, 0, 0])
            else:
                total_list.append([scores, row, col, gene1, gene2, 0, 0, 1])

        print('Completed conversion.')
    else:
        print('Train labels are excluded.')
        for i in score_sort_toplist:
            scores = i[0]
            row = i[1]
            gene1 = node_names[row]
            col = i[2]
            gene2 = node_names[col]
            prediction_label_pair = (row, col)
            if prediction_label_pair in test_label_pairs:
                total_list.append([scores, row, col, gene1, gene2, 0, 1, 0])
            else:
                total_list.append([scores, row, col, gene1, gene2, 0, 0, 1])
        print('Completed conversion.')


def process_table(rows, cols, gene1, gene2, scores, train_edge, test_edge, new_edge):
    print('\n== Process curated prediction score to build a table ==')
    table = pd.DataFrame()
    table['row'] = pd.Series(rows)
    table['col'] = pd.Series(cols)
    table['gene1'] = pd.Series(gene1)
    table['gene2'] = pd.Series(gene2)
    table['score'] = pd.Series(scores)
    table['train_edge'] = pd.Series(train_edge)
    table['test_edge'] = pd.Series(test_edge)
    table['new_edge'] = pd.Series(new_edge)
    print('#table shape: ', table.shape)
    table = table.assign(score_ranking=len(table.score)-stats.rankdata(table.score, method='max')+1)
    print('\nsort the table with score-descending order...')
    table_sort_score = table.sort_values(by='score', ascending=False)
    table_sort_score = table_sort_score[['row', 'col', 'gene1', 'gene2', 'score', 'score_ranking', 'train_edge',
                                         'test_edge', 'new_edge']]
    print(f'#final table shape: {table.shape}\n'
          f'Completed processing to build a table.')
    return table_sort_score


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, help="input result: gcn_cv.jbl")
    parser.add_argument('--dataset', type=str, help="input dataset: dataset.jbl")
    parser.add_argument('--node', type=str, help="input dataset node: dataset_node.csv")
    parser.add_argument('--cv', default=0, type=int, help="cross validation: select 0,1,2,3,4")
    parser.add_argument('--output', type=str, help="output:score.txt")
    parser.add_argument('--scorerank', default=10000, type=int, help='pick score ranking from 1 to score_cutoff_value')
    parser.add_argument("-t", '--train', action="store_true", help="default: exclude train label at score ranking list")
    parser.add_argument("-n", "--proc_num", type=int, default=1, help="a number of processors for multiprocessing.")
    args = parser.parse_args()
    print(f'\n== args summary ==\n'
          f'args result: {args.result}\n'
          f'args dataset: {args.dataset}\n'
          f'args node: {args.node}\n'
          f'args cv: {args.cv}\n'
          f'args output: {args.output}\n'
          f'args score rank: {args.scorerank}\n'
          f'args train: {args.train}\n'
          f'args proc num: {args.proc_num}')
    return args


def split_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]


def main():
    args = get_parser()
    start_time = time.time()

    node_names = build_node_name(args.node)
    with open("./test_label_pairs.pkl", "rb") as f:
        # build test label pairs
        # test_label_pairs = build_test_label_pairs(args.result,args.cv) # need to stay
        # f2 = open('./test_label_pairs.pkl', 'wb')
        # pickle.dump(test_label_pairs, f2)
        test_label_pairs = pickle.load(f)
    # build all prediction target pairs
    target_label_pairs = build_target_label_pairs(args.dataset)
    train_label_pairs = list(set(target_label_pairs) - set(test_label_pairs))

    print(f'\n== Summary of edge label data ==\n'
          f'#target_label_pairs: {len(target_label_pairs)}\n'
          f'#train_label_pairs: {len(train_label_pairs)}\n'
          f'#test_label_pairs: {len(test_label_pairs)}')
    score_sort_toplist = sort_prediction_score(args.result, args.cv, target_label_pairs, test_label_pairs,
                                               args.scorerank, args.train)
    # convert score for dataframe
    n_proc = args.proc_num
    pool = Pool(processes=n_proc)
    split_score_sort_toplist = list(split_list(score_sort_toplist, n_proc))
    with Manager() as manager:
        total_list = manager.list()
        convert_ = partial(convert, target_label_pairs=target_label_pairs, test_label_pairs=test_label_pairs,
                           node_names=node_names, train=args.train, total_list=total_list)
        pool.map(convert_, split_score_sort_toplist)
        scores = [l[0] for l in total_list]
        rows = [l[1] for l in total_list]
        cols = [l[2] for l in total_list]
        gene1 = [l[3] for l in total_list]
        gene2 = [l[4] for l in total_list]
        train_edge = [l[5] for l in total_list]
        test_edge = [l[6] for l in total_list]
        new_edge = [l[7] for l in total_list]
        print(f'\n#rows: {len(rows)}\n'
              f'#cols: {len(cols)}\n'
              f'#gene1: {len(gene1)}\n'
              f'#gene2: {len(gene2)}\n'
              f'#scores: {len(scores)}\n'
              f'#train_edge: {len(train_edge)}\n'
              f'#test_edge: {len(test_edge)}\n'
              f'#new_edge: {len(new_edge)}')

        table_sort_score = process_table(rows, cols, gene1, gene2, scores, train_edge, test_edge, new_edge)
        print(f'\n== Export the processed result as txt file ==\n'
              f'output file path: {args.output}')
        with open(args.output, 'w') as f:
            table_sort_score.to_csv(f, sep='\t', header=True, index=False)

        elapsed_time = time.time() - start_time
        print(f'\n#time:{elapsed_time} sec\n'
              f'-- fin --\n')


if __name__ == '__main__':
    main()
