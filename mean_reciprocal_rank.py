
# Author: yoshi
# Date: 6/12/2019
# Updated: 

## Script for calculate mean reciprocal rank (MRR)

# Usage: python mean_reciprocal_rank.py --input score_ip_cv0.txt --cv 0

import numpy as np
import time
import argparse

# set start time
start_time=time.time()

# set argparse
parser=argparse.ArgumentParser()
parser.add_argument('--input',type=str) 
parser.add_argument('--cv',default=0,type=int) #cross validation: select 0,1,2,3,4
args=parser.parse_args()

#print('args result='+args.result)
print('cross validation fold: cv={0}'.format(args.cv))

# Mean Reciprocal Rank function (input: list of score ranking that test_edge represents as "1")
def mrr(input_list):
	input_np=np.array(input_list, dtype='float')
	reciprocal=np.reciprocal(input_np)
	mean_reciprocal_rank=np.sum(reciprocal)/len(input_np)
	mrr=mean_reciprocal_rank.tolist()
	return mrr

# 
print('Load data... ')
print('load: {0}'.format(args.input))
test_edge_score_rank=[]
with open(args.input, 'r') as f:
	next(f)
	for line in f:
		l=line.strip().split('\t')
		score_rank=l[5]
		test_edge=l[8]
		if int(test_edge)==1:
			test_edge_score_rank.append(score_rank)
		else:
			pass


mrr=mrr(test_edge_score_rank)

print('\nresult of mrr calculation...')
print('cv fold: {0}'.format(args.cv))
print('mrr: {0}'.format(mrr))


# measure time
elapsed_time=time.time() - start_time
print('\n#time:{0}'.format(elapsed_time)+' sec')


print('-- fin --\n')


