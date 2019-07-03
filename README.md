# Program for convertion of prediction score on ip/distmult/gcn

Prep  
1. You need to put three pickle data below  
`node_names.pkl`
`test_label_pairs.pkl`
`target_label_pairs.pkl`

Run  
`python prediction_score.py --result ./test_score_matrix_10000.pkl --dataset ./dataset.jbl --node ./dataset_node.csv --cv 0 --scorerank 15000 --train true --output ./score.txt`

You can change input result file whatever you want to try.  
`--result ./test_score_matrix_10000.pkl`

data  
