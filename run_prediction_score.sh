#!/bin/sh

# Date: 6/11/2019
# Update: 7/26/2019
# Usage: sh run_predidtion_score.sh ip2/distmult2/gcn2

cd `dirname $0`

cv="0 1 2 3 4"
method=$1
echo "method: $method"

for i in $cv; do
	echo "cv fold: $i"
	python prediction_score_for_multiprocess.py --result ./result/main/target6/cv_info.${method}.jbl --dataset ./processed_data/target6/dataset.jbl --node ./processed_data/target6/dataset_node.csv --cv $i --score_rank 20000000 --cutoff 20000000 --train -n 2 --output ./result/main/target6/score_${method}_cv${i}.txt --output_pkl ./result/main/target6/score_${method}_cv${i}.pkl --edge_type ppi
done

