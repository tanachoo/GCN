"""
File name: prediction_score_for_multiprocess.py
Author: yoshi, shoichi
Description: Script for converting prediction score to table
Date: 15 July 2019
"""


import argparse
from functools import partial
from multiprocessing import Pool, Manager
import pickle
import pprint
import time

import joblib
import pandas as pd
from scipy import stats


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
          f'load: {filename}')  # node data='dataset_node.csv'
    with open(filename, 'r') as f:
        node_names = [l.strip() for l in f]
    print(f'#node_names: {len(node_names)}')
    return node_names


def build_test_label_pairs(filename, cv):
    """ To make test label pair list """
    # import main result data (post caluculation jbl file)
    print(f'\n== Prep test label pairs list ==\n'
          f'load: {filename}\n'
          f'cv fold: {cv}')
    result_data = joblib.load(filename)
    test_labels = result_data[cv]['test_labels']
    test_label_pairs = []

    for i in test_labels[0]:
        test_label_pair = (i[0], i[2])
        test_label_pair = tuple(sorted(test_label_pair))
        test_label_pairs.append(test_label_pair)

    print(f'#test_label_pairs: {len(test_label_pairs)}\n'
          f'Remove duplicate.')
    test_label_pairs = list(set(test_label_pairs))  # remove duplicated in list of test_label_pairs
    print(f'#duplicate-removed test_label_pairs: {len(test_label_pairs)}\n'
          f'Completed to prep test label list.')
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
          f'Remove duplicate.')
    target_label_pairs = list(set(target_label_pairs))  # remove duplicated in list of target_label_pairs
    print(f'#duplicate-removed target_label_pairs: {len(target_label_pairs)}\n'
          f'Completed to prep target label list.')
    return target_label_pairs


def sort_prediction_score(filename, cv, target_label_pairs, test_label_pairs, scorerank, cutoff, train, edgetype):
    """ Sort prediction result array matrix and Set threshold """
    print('\n== Sort predisction score ==')
    print(f'load: {filename}')
    with open(filename, 'rb') as f:  # only activate when test sample data
        result_data = pickle.load(f)  # only activate when test sample data
    # result_data = joblib.load(filename)
    print(f'cv fold: {cv}')
    # prediction = result_data[cv]['prediction_data']
    # matrix = prediction[0]
    matrix = result_data  # only activate when test sample data
    print(f'prediction score matrix shape: {matrix.shape}\n'
          f'\nPrep list of [(score,row,col)] from prediction score results matrix.')
    dim_row = matrix.shape[0]
    dim_col = matrix.shape[1]
    score_row_col = [(matrix[row, col], row, col) for row in range(dim_row) for col in range(row+1, dim_col)]
    print(f'#scores as adopted: {len(score_row_col)}')  # should be 480577503

    if edgetype == 'ppi':
        """ protein-protein """
        print(f'Pick protein-protein interaction.')
        ppi1 = [i for i in score_row_col if i[1] < 3071 or i[1] > 14506]
        ppi = [i for i in ppi1 if i[2] < 3071 or i[2] > 14506]
        print(f'#total protein-protein edge: {len(ppi)}\n') # should be 191423961
        edgetype_selection_score = ppi

    elif edgetype == 'pci':
        """ protein-chemical """
        print(f'Pick protein-chemical interaction.')
        pci1 = [i for i in score_row_col if i[1] < 3071 and 3070 < i[2] < 14507]
        pci2 = [i for i in score_row_col if 3070 < i[1] < 14507 and 14506 < i[2] < 31003]
        pci = pci1 + pci2
        print(f'#total protein-chemical edge: {len(pci)}\n') # should be 223768212
        edgetype_selection_score = pci

    elif edgetype == 'cci':
        """ chemical-chemical """ 
        print(f'Pick chemical-chemical interaction.')
        cci = [i for i in score_row_col if 3070 < i[1] < 14507 and 3070 < i[2] < 14507]
        print(f'#total chemical-chemical edge: {len(cci)}\n') # should be 65385330
        edgetype_selection_score = cci

    # sort scores with descending order
    print('Sort scores and pre-pick toplist by cutoff value.')
    edgetype_selection_score.sort(reverse=True)  # Sort list based on "score" with a decending order
    score_sort = edgetype_selection_score[:cutoff]  # args.cutoff: Pick top list using arbitrary threshold
    print(f'#pre-picked top score list: {len(score_sort)}')

    if train:
        print(f'(Train labels are included for preparing score-ordred list.)\n'
              f'Pick toplist by scorerank.')
        score_sort_toplist = score_sort[:scorerank]  # args.scorerank: Select top score ranking to export
        print(f'#score post pick score-rank: {len(score_sort_toplist)}\n'
              f'Completed to prep prediction score-ordered list including train labels.')
        return score_sort_toplist
    else:
        print(f'(Train labels are excluded for preparing score-ordred list.)\n'
              f'Pick toplist by scorerank.')
        train_label_pairs = list(set(target_label_pairs) - set(test_label_pairs)) # Prep target,test,train label list
        score_tmp = [i for i in score_sort if (i[1], i[2]) not in set(train_label_pairs)]
        score_tmp.sort(reverse=True)
        score_sort_toplist = score_tmp[:scorerank]
        print(f'#score post pick score-rank: {len(score_sort_toplist)}\n'
              f'Completed to prep prediction score-ordered list w/o train labels.')
        return score_sort_toplist


def convert(score_sort_toplist, target_label_pairs, test_label_pairs, node_names, train, total_list):
    """
    let score-sorted list [(score,row,col)...] convert to table
    total_list = (scores, rows, cols, gene1, gene2, train_edge, test_edge, new_edge)
    """
    tmp_list = []
    if train:
        for i in score_sort_toplist:
            scores = i[0]
            row = i[1]
            gene1 = node_names[row]
            col = i[2]
            gene2 = node_names[col]
            prediction_label_pair = (row, col)
            if prediction_label_pair in target_label_pairs:
                if prediction_label_pair in test_label_pairs:
                    tmp_list.append([scores, row, col, gene1, gene2, 0, 1, 0])
                else:
                    tmp_list.append([scores, row, col, gene1, gene2, 1, 0, 0])
            else:
                tmp_list.append([scores, row, col, gene1, gene2, 0, 0, 1])
    else:
        for i in score_sort_toplist:
            scores = i[0]
            row = i[1]
            gene1 = node_names[row]
            col = i[2]
            gene2 = node_names[col]
            prediction_label_pair = (row, col)
            if prediction_label_pair in test_label_pairs:
                tmp_list.append([scores, row, col, gene1, gene2, 0, 1, 0])
            else:
                tmp_list.append([scores, row, col, gene1, gene2, 0, 0, 1])
    total_list.extend(tmp_list)


def process_table(rows, cols, gene1, gene2, scores, train_edge, test_edge, new_edge):
    """ To build a table """
    print('\n== Process curated prediction score to build a table ==')
    table = pd.DataFrame({
        "row": rows,
        "col": cols,
        "gene1": gene1,
        "gene2": gene2,
        "score": scores,
        "train_edge": train_edge,
        "test_edge": test_edge,
        "new_edge": new_edge
    })
    # print('#table shape: ', table.shape)
    table = table.assign(score_ranking=len(table.score) - stats.rankdata(table.score, method='max') + 1)
    print('Sort the table with score-descending order.')
    table_sort_score = table.sort_values(by='score', ascending=False)
    table_sort_score = table_sort_score[['row', 'col', 'gene1', 'gene2', 'score', 'score_ranking', 'train_edge',
                                         'test_edge', 'new_edge']]
    print(f'#final table shape: {table.shape}\n'
          f'Completed processing to build a table.')
    return table_sort_score


def enrichment(target_label_pairs, test_label_pairs, table_sort_score, cv, train, edgetype):
    print('\n== Calculate enrichment ==')
    train_label_pairs = list(set(target_label_pairs) - set(test_label_pairs)) # prep train edges list

    if train:
        if edgetype == 'ppi':
            total = 191423961
        elif edgetype == 'pci':
            total = 223768212
        elif edgetype == 'cci':      
            total = 65385330

        total_wo_train = total - len(train_label_pairs)  # remove train edges from total
        total_test_edges = len(test_label_pairs)
        table_wo_train = table_sort_score[table_sort_score.train_edge == 0]  # prep table w/o train edges (remove train from the table)
        print(f'Summary of edges attribution\n'
              f'cv fold: {cv}\n'
              f'#total as scored: {total}\n'
              f'#total_w/o_train_edges: {total_wo_train}\n'
              f'#total_target_edges: {len(target_label_pairs)}\n'
              f'#total_train_edges: {len(train_label_pairs)}\n'
              f'#total_test_edges: {len(test_label_pairs)}\n')

        # enrichment calcucation
        top = [0.1, 0.5, 1.0]  # top: 0.1%, 0.5%, 1%, 3%, 5%
        for i in top:
            ratio = i*0.01
            top_ratio = round(total_wo_train*ratio)  # calculate the number of top list based on top%
            table_wo_train_toplist = table_wo_train.iloc[:top_ratio, ]  # pick top list from the table w/o train edges
            test_edges_in_toplist = len(table_wo_train_toplist[table_wo_train_toplist.test_edge == 1].index)
            test_edges_enrichment = test_edges_in_toplist/total_test_edges
            print(f'#top%: {i}\n'
                  f'#top_ratio: {top_ratio}\n'
                  f'#test_edges_in_toplist: {test_edges_in_toplist}\n'
                  f'#test edges enrichment top{i}%: {test_edges_enrichment}\n')

    else:
        pass # built later...


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, help="input result: gcn_cv.jbl")
    parser.add_argument('--dataset', type=str, help="input dataset: dataset.jbl")
    parser.add_argument('--node', type=str, help="input dataset node: dataset_node.csv")
    parser.add_argument('--cv', default=0, type=int, help="cross validation: select 0,1,2,3,4")
    parser.add_argument('--output', type=str, help="output:score.txt")
    parser.add_argument('--scorerank', default=10000, type=int, help='pick score ranking from 1 to scorerank')
    parser.add_argument('--cutoff', default=10000, type=int, help='pre-pick score ranking from 1 to cutoff, should cutoff > scorerank')
    parser.add_argument("-t", '--train', action="store_true", help="default: exclude train label at score ranking list")
    parser.add_argument("-n", "--proc_num", type=int, default=1, help="a number of processors for multiprocessing.")
    parser.add_argument('--edgetype', type=str, help="edgetype: ppi(protein-protein), pci(protein-chemical), cci(chemical-chemical)")
    args = parser.parse_args()

    print('\n== args summary ==')
    pprint.pprint(vars(args))
    return args


def split_list(l, n):
    return [l[i::n] for i in range(n)]


def main():
    args = get_parser()
    start_time = time.time()

    node_names = build_node_name(args.node)
    # test_label_pairs = build_test_label_pairs(args.result, args.cv) # main code
    with open("./test_label_pairs.pkl", "rb") as f:  # only activate when test sample data
        test_label_pairs = pickle.load(f)  # only activate when test sample data
    target_label_pairs = build_target_label_pairs(args.dataset)
    score_sort_toplist = sort_prediction_score(args.result, args.cv, target_label_pairs, test_label_pairs,
                                               args.scorerank, args.cutoff, args.train, args.edgetype)

    print('\n== Start convesion of prediction scores ==')
    print(f'Train labels are {["included" if args.train else "excluded"][0]}.')
    n_proc = args.proc_num
    pool = Pool(processes=n_proc)
    split_score_sort_toplist = split_list(score_sort_toplist, n_proc)
    with Manager() as manager:
        total_list = manager.list()
        convert_ = partial(convert, target_label_pairs=set(target_label_pairs), test_label_pairs=set(test_label_pairs),
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
        print('Completed conversion.')

    table_sort_score = process_table(rows, cols, gene1, gene2, scores, train_edge, test_edge, new_edge)
    print(f'\n== Export the processed result as txt file ==\n'
          f'output file path: {args.output}')
    with open(args.output, 'w') as f:
        table_sort_score.to_csv(f, sep='\t', header=True, index=False)

    enrichment(target_label_pairs, test_label_pairs, table_sort_score, args.cv, args.train, args.edgetype)

    elapsed_time = time.time() - start_time
    print(f'\n#time:{elapsed_time} sec\n'
          f'-- fin --\n')


if __name__ == '__main__':
    main()
