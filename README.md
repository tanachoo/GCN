# Program for convertion of prediction score on ip/distmult/gcn

Prep  
1. You need to put three pickle data  
`node_names.pkl`
`test_label_pairs.pkl`
`target_label_pairs.pkl`  
2. Select result test data  
`test_score_matrix_10000.pkl` (matrix:100×100)  
`test_score_matrix_90000.pkl` (matrix:300×300)  
`test_score_matrix_250000.pkl` (matrix:500×500)  
`test_score_matrix_1000000.pkl` (matrix:1000×1000)  

Run  
`python prediction_score_spare2.py --result ./test_score_matrix_10000.pkl --dataset ./dataset.jbl --node ./dataset_node.csv --cv 0 --scorerank 15000 --train true --output ./score.txt`

You can change input result file whatever you want to try.  
`--result ./test_score_matrix_10000.pkl`

data  
