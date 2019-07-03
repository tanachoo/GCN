# Author: yoshi
# Write start: 5/13/2019
# Updated: 7/3/2019

## Script for converting prediction score to table

# Usage: python prediction_score.py --result ./pool_result/target0_20190425/cv_info.gcn.jbl --dataset ./processed_data/target0_20190425/dataset.jbl \
# --node ./processed_data/target0_20190425/dataset_node.csv --cv 0 --scorerank 15000 --train true --output ./pool_result/target0_20190425/score.txt

import joblib
import numpy as np
import pandas as pd
import time
import argparse
from scipy import stats
from multiprocessing import Pool
import multiprocessing as mp
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

    print('\n== Prep node names list ==')
    print('load: {0}'.format(filename)) #node data='../../processed_data/target0_20190425/dataset_node.csv'

    node_names=[]

    with open(filename,'r') as f:
        for line in f:
            nodes=line.strip()
            node_names.append(nodes)

    print('#node_names: ',len(node_names))

    return node_names

def build_test_label_pairs(filename,cv):
    """ To make test label pair list """

    # import main result data (post caluculation jbl file)
    print('\n== Prep test label pairs list ==')
    print('load: {0}'.format(filename))
    result_data=joblib.load(filename)
    print('cv fold: {0}'.format(cv)) 

    test_labels=result_data[cv]['test_labels']

    test_label_pairs=[] # store test-label-pair (tuple) as list

    for i in test_labels[0]:
        test_label_pair=(i[0],i[2])
        test_label_pair=tuple(sorted(test_label_pair))
        test_label_pairs.append(test_label_pair)

    print('#test_label_pairs: ',len(test_label_pairs))
    print('#original_test_label: ',len(test_labels[0]))
    print('(should be same values...)')

    print('\nremove duplicate...')
    test_label_pairs=list(set(test_label_pairs)) # remove duplicated in list of test_label_pairs
    print('#duplicate removed test_label_pairs: ',len(test_label_pairs))

    print('okay... Completed to prep test label list.')

    return test_label_pairs

def build_target_label_pairs(filename): #args.dataset (input data jbl file)
    """To make all prediction target (train+test) label pair list"""
    # import all edge label data (input data for establish model, train + test) 
    print('\n== Prep all target label pairs list ==')
    print('load: {0}'.format(filename))
    input_data=joblib.load(filename)
    label_list=input_data['label_list']

    target_label_pairs=[] # store all target-label-pair (tuple) as list

    for i in label_list[0]:
        label_pair=(i[0],i[2])
        label_pair=tuple(sorted(label_pair))
        target_label_pairs.append(label_pair)

    print('#target_label_pairs: ',len(target_label_pairs))
    print('#original_label_pairs: ',len(label_list[0]))
    print('(should be same values...)')

    print('\nremove duplicate...')
    target_label_pairs=list(set(target_label_pairs)) # remove duplicated in list of target_label_pairs
    print('#duplicated removed target_label_pairs: ',len(target_label_pairs))

    print('okay... Completed to prep target label list.')

    return target_label_pairs
    
def sort_prediction_score(filename,cv,target_label_pairs,test_label_pairs,scorerank,train):
    """ Sort prediction result array matrix and Set threshold """
    print('\n== Sort predisction score ==')

    ## Load data
    print('load: {0}'.format(filename))
    f = open(filename,'rb') # add for test
    result_data=pickle.load(f) # add for test
    #result_data=joblib.load(filename)
    print('cv fold: {0}'.format(cv))
    #prediction=result_data[cv]['prediction_data']
    #matrix=prediction[0]
    matrix=result_data # add for test
    print('prediction score matrix shape: ',matrix.shape)

    print('\nprep list of [(score,row,col),(),(),,,,] from prediction score results matrix...')
    score_row_col=[]
    dim_row=matrix.shape[0]
    dim_col=matrix.shape[1]
    ## Store prediction score as a set of (score,row,col) tuple
    score_row_col=[(matrix[row,col],row,col) for row in range(dim_row) for col in range(row+1, dim_col)]

    print('#scores as adopted: ',len(score_row_col)) # 480577503
    #print('(should be 480577503...)')

    ## Here need to delete CHEBI ID
    ## remove row CHEBI
    row_CHEBI_deleted=[i for i in score_row_col if i[1]<3071 or i[1]>14506]
    ## remove col CHEBI
    row_col_CHEBI_deleted=[i for i in row_CHEBI_deleted if i[2]<3071 or i[2]>14506]
    print('#scores as adopted post removal of CHEBI nodes: ',len(row_col_CHEBI_deleted))
    
    ## sort scores with descending order
    print('\nSort scores and Pick top 2000000...')
    row_col_CHEBI_deleted.sort(reverse=True) # Sort list based on "score" with a decending order
    score_sort=row_col_CHEBI_deleted[:2000000] # Cut top list using arbitrary threshold
    #score_sort=sorted(row_col_CHEBI_deleted,reverse=True) # this "sorted" method is slower
    #score_sort=score_sort[:3000000]
    print('okay...done.')

    # Prep target,test,train label list
    target_label_pairs=target_label_pairs
    test_label_pairs=test_label_pairs
    train_label_pairs=list(set(target_label_pairs) - set(test_label_pairs))
    
    if train=='true':
        print('\nTrain labels are included for preparing score-ordred list.')
        print('#scores including train labels: ',len(score_sort))
        print('Cutoff top list by score-rank...')
        score_sort_toplist=score_sort[:scorerank] #args.scorerank: Select top score ranking to export
        print('score rank cutoff value: {0}'.format(scorerank)) # should be less than 3,000,000
        print('#score post score-rank cutoff: ',len(score_sort_toplist))
        print('(should be same values...)')

        print('Completed to prep prediction score-ordered list including train labels.')

        return score_sort_toplist
    
    else:
        print('\nTrain labels are excluded for preparing score-ordred list.')

        score_tmp=[]
        score_tmp=[i for i in score_sort if (i[1],i[2]) not in train_label_pairs]

        ## sort with a decending order
        score_tmp.sort(reverse=True)
        print('#scores post removal of train labels: ',len(score_tmp))
        score_sort_toplist=score_tmp[:scorerank]
        print('score rank cutoff value: {0}'.format(scorerank))
        print('#src_score_sort_toplist: ',len(score_sort_toplist))
        print('(should be same values...)')

        print('Completed to prep prediction score-ordered list w/o train labels.')
        
        return score_sort_toplist
        
 def convert(score_sort_toplist,target_label_pairs,test_label_pairs,node_names,train):
    """ let score-sorted list [(score,row,col),...] convert to table """
    print('\n== Start convesion of prediction scores ==')

    scores=[]
    rows=[]
    cols=[]
    gene1=[]
    gene2=[]
    train_edge=[]
    test_edge=[]
    new_edge=[]

    node_names=node_names
    target_label_pairs=target_label_pairs
    test_label_pairs=test_label_pairs
    train_label_pairs=list(set(target_label_pairs) - set(test_label_pairs))

    if train=='true':
        print('Train labels are included.')

        scores=[i[0] for i in score_sort_toplist]        
        rows=[i[1] for i in score_sort_toplist]
        cols=[i[2] for i in score_sort_toplist]
        gene1=[node_names[i[1]] for i in score_sort_toplist]
        gene2=[node_names[i[2]] for i in score_sort_toplist]

        for i in score_sort_toplist:
            prediction_label_pair=(i[1],i[2])

            if prediction_label_pair in target_label_pairs: # To see if "prefiction label pair" is in a set of "target label pairs"
                new_edge.append(0) # If it's true, add "0" at new_edge, which means that "prediction label pair" is not new edge. Otherwise, add "1" at new_edge.
                if prediction_label_pair in test_label_pairs: # And, to see if "predoction_label_pair" is in a set of "test label pairs"
                    test_edge.append(1) # If it's ture, add "1" at test_edge and "0" at train_edge
                    train_edge.append(0)
                else:
                    test_edge.append(0) # Oherwise, add "0" at test_edge and "1" at train_edge
                    train_edge.append(1)
            else:
                train_edge.append(0)
                test_edge.append(0)
                new_edge.append(1)
                
        print('Completed conversion.')
        return rows,cols,gene1,gene2,scores,train_edge,test_edge,new_edge
        
    else:
        print('Train labels are excluded.')

        scores=[i[0] for i in score_sort_toplist]        
        rows=[i[1] for i in score_sort_toplist]
        cols=[i[2] for i in score_sort_toplist]
        gene1=[node_names[i[1]] for i in score_sort_toplist]
        gene2=[node_names[i[2]] for i in score_sort_toplist]
        train_edge=[0 for i in score_sort_toplist]

        for i in score_sort_toplist:
            prediction_label_pair=(i[1],i[2])

            if prediction_label_pair in test_label_pairs: # To see if "prediction label pair" is in a set "prediction_label_pair"
                test_edge.append(1) # If it's ture, add "1" at test_edge and "0" at new_edge
                new_edge.append(0)
            else:
                test_edge.append(0)
                new_edge.append(1)

        print('Completed conversion.')
        return rows,cols,gene1,gene2,scores,train_edge,test_edge,new_edge
        
def process_table(rows,cols,gene1,gene2,scores,train_edge,test_edge,new_edge):
    # Prep pandas dataframe section 
    print('\n== Process curated prediction score to build a table ==')
    table=pd.DataFrame()

    # Add columns to dataframe table
    table['row']=rows
    table['col']=cols
    table['gene1']=gene1
    table['gene2']=gene2
    table['score']=scores
    table['train_edge']=train_edge
    table['test_edge']=test_edge
    table['new_edge']=new_edge
    print('#table shape: ',table.shape)

    # assign score ranking
    table=table.assign(score_ranking=len(table.score)-stats.rankdata(table.score, method='max')+1)

    # sort table with high score order
    print('\nsort the table with score-descending order...')
    table_sort_score=table.sort_values(by='score',ascending=False)

    # change column order
    table_sort_score=table_sort_score[['row','col','gene1','gene2','score','score_ranking','train_edge','test_edge','new_edge']]
    
    print('#final table shape: ',table.shape)
    print('Completed processing to build a table.')

    return table_sort_score
    
if __name__ == '__main__':

    # set argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--result',type=str) #input result: gcn_cv.jbl
    parser.add_argument('--dataset',type=str) #input dataset: dataset.jbl
    parser.add_argument('--node',type=str) #input dataset node: dataset_node.csv
    parser.add_argument('--cv',default=0,type=int) #cross validation: select 0,1,2,3,4
    parser.add_argument('--output',type=str) # output:score.txt
    parser.add_argument('--scorerank',default=10000,type=int) # Select score ranking from 1 ~ score_cutoff_value
    parser.add_argument('--train',default='false',type=str) # default: exclude train label at score ranking list
    args=parser.parse_args()

    print('\n== args summary ==')
    print('args result: '+args.result)
    print('args dataset: '+args.dataset)
    print('args node: '+args.node)
    print('args cv: {0}'.format(args.cv))
    print('args output: '+args.output)
    print('args score rank: {0}'.format(args.scorerank))
    print('args train: '+args.train)

    # set start time
    start_time=time.time()


    # node names
    #node_names=build_node_name(args.node)
    #f1 = open('./node_names.pkl', 'wb')
    #pickle.dump(node_names, f1)
    
    o1 = open('./node_names.pkl','rb')
    node_names = pickle.load(o1)
    #print(len(node_names))
    
    # build test label pairs
    #test_label_pairs=build_test_label_pairs(args.result,args.cv)
    #f2 = open('./test_label_pairs.pkl', 'wb')
    #pickle.dump(test_label_pairs, f2)
    
    o2 = open('./test_label_pairs.pkl','rb')
    test_label_pairs = pickle.load(o2)
    #print(len(test_label_pairs))    

    # build all prediction target pairs
    #target_label_pairs=build_target_label_pairs(args.dataset)
    #f3 = open('./target_label_pairs.pkl', 'wb')
    #pickle.dump(target_label_pairs, f3)
    
    o3 = open('./target_label_pairs.pkl','rb')
    target_label_pairs = pickle.load(o3)
    #print(len(target_label_pairs))
    
    # train label pair
    train_label_pairs=list(set(target_label_pairs) - set(test_label_pairs))

    # Edge attribution summary
    print('\n== Summary of edge label data ==')
    print('#target_label_pairs: ',len(target_label_pairs))
    print('#train_label_pairs: ',len(train_label_pairs))
    print('#test_label_pairs: ',len(test_label_pairs))
    
    # sort with predisction score
    score_sort_toplist=sort_prediction_score(args.result,args.cv,target_label_pairs,test_label_pairs,args.scorerank,args.train)

    # convert score for dataframe
    rows,cols,gene1,gene2,scores,train_edge,test_edge,new_edge = convert(score_sort_toplist,target_label_pairs,test_label_pairs,node_names,args.train)
    
    print('\n#rows: ',len(rows))
    print('#cols: ',len(cols))
    print('#gene1: ',len(gene1))
    print('#gene2: ',len(gene2))
    print('#scores: ',len(scores))
    print('#train_edge: ',len(train_edge))
    print('#test_edge: ',len(test_edge))
    print('#new_edge: ',len(new_edge))
    
    # process table
    table_sort_score = process_table(rows,cols,gene1,gene2,scores,train_edge,test_edge,new_edge)
    
    # write table
    print('\n== Export the processed result as txt file ==')
    print('output file path: '+args.output)
    with open(args.output, 'w') as f:
        table_sort_score.to_csv(f,sep='\t',header=True,index=False)
    
    # measure time
    elapsed_time=time.time() - start_time
    print('\n#time:{0}'.format(elapsed_time)+' sec')

    print('-- fin --\n')
    
    
    
