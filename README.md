# GCN analysis Program for convertion of link prediction score on ip/distmult/gcn


## TEST
1. Prepare three data below on a same directory where the main program (prediction_score_for_multiprocess.py) stays.  
`dataset.jbl (node_names.pkl for test)`
`dataset_node.csv (test_label_pairs.pkl for test)`
`target_label_pairs.pkl`  

2. Select result test data  
`test_score_matrix_10000.pkl` (matrix:100×100, approximately 20 sec to complete this script)  
`test_score_matrix_90000.pkl` (matrix:300×300, approximately 3 min to complete this script)  
`test_score_matrix_250000.pkl` (matrix:500×500)  
`test_score_matrix_1000000.pkl` (matrix:1000×1000, approximately 30 min to complete this script)  

3. Run  
`python prediction_score_for_multiprocess.py --result ./test_score_matrix_10000.pkl --dataset ./dataset.jbl --node ./dataset_node.csv --cv 0 --score_rank 1000000 --cutoff 1000000 --train --output ./score.txt --output_pkl ./score.pkl -n 1 --edge_type ppi`

You can change input results file whatever you want to try from above four types of dataset.  
`--result ./test_score_matrix_*10000*.pkl`

Result (example of output file)  
See: `sample_output.txt`

## MAIN
1. Prepare  
prediction data (--result): `cv_info.ip2.jbl` `cv_info.distmult2.jbl` `cv_info.gcn2.jbl`   
input data (--dataset): `dataset.jbl`   
node data (--node): `dataset_node.csv`  

2. Set args  
--cv: cross validation  
--score_rank: 20000000 (top 10%), 10000000(top 5%)  
--cut_off:  20000000 (top 10%), 10000000(top 5%)  
--train: include train label  
--output: set txt output file name and path  
--output_pkl: set pkl output file name and path  
-n: cpu  
--edge_type: ppi(protein protein interaction), pci(protein chemical interaction), cci(chemical chemical interaction)  

3. Run  
`sh run_prediction_score.sh`  
