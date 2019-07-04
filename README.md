# Program for convertion of prediction score on ip/distmult/gcn

Prepare
1. You need to put three pickle data on a same directory where the main program(prediction_score_spare2.py) stays.  
`node_names.pkl`
`test_label_pairs.pkl`
`target_label_pairs.pkl`  

2. Select result test data  
`test_score_matrix_10000.pkl` (matrix:100×100, approximately 20 sec to complete this script)  
`test_score_matrix_90000.pkl` (matrix:300×300, approximately 3 min to complete this script)  
`test_score_matrix_250000.pkl` (matrix:500×500)  
`test_score_matrix_1000000.pkl` (matrix:1000×1000, approximately 30 min to complete this script)  

Run (Use: prediction_score_spare2.py)  
`python prediction_score_spare2.py --result ./test_score_matrix_10000.pkl --dataset ./dataset.jbl --node ./dataset_node.csv --cv 0 --scorerank 1000000 --train true --output ./score.txt`

(Ignore: `--dataset, --node, --cv, --scorerank, --train` (keep as they are))

You can change input results file whatever you want to try from above four types of dataset.  
`--result ./test_score_matrix_*10000*.pkl`

NOTE: You just change below two args to run. Please stay other args commands as above.  
`--result`  
`--output` (output file name)

Result (output file)  
See: `sample_output.txt`
