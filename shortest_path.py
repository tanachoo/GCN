
# Auther: yoshi
# Date: 9/2/2019
# Updated: 9/9/2019
# Project: gcn pathaway

## script for calculate shortest path in Graph
##Usage: python shortest_path.py --input ../result/main/ppi/target1/score_ip2_cv0.pkl --graph ./wo_target1.ppigraph.tsv --output ./score_path_ppi_target1_cv0_ip2.txt

import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import time

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help="input: score_gcn2_cv0.pkl")
    parser.add_argument('--graph', type=str, help="graph data: tsv file")
    parser.add_argument('--output', type=str, help="output(filetype=txt): score.txt")
    args = parser.parse_args()


    ## load data
    # pkl data: score_gcn2_cv0.pkl
    with open(args.input, 'rb') as f:
        table = pickle.load(f)
    print(f'#table shape: {table.shape}\n')
    # print(data.columns)
    gene1 = table['gene1'].values.tolist()
    gene2 = table['gene2'].values.tolist()

    ## build graph
    # input_graph = './test_graph_input.tsv' (format: node tab node)
    G = nx.read_edgelist(args.graph, nodetype=str, create_using=nx.MultiGraph())
    print(f'load graph: {args.graph}\n'
          f'number_of_nodes: {nx.number_of_nodes(G)}\n'
          f'number_of_edges: {nx.number_of_edges(G)}')

    ## calculate shortest path
    shortest_path = []
    for i in range(len(gene1)):
        source = gene1[i]
        target = gene2[i]
        try:
            shortest_path_length = nx.shortest_path_length(G, source=source, target=target)
            shortest_path.append(shortest_path_length)
        except:
            shortest_path.append(0)
    print(f'#shortest_path: {len(shortest_path)}')

    ## Insert shortest path to the table
    table = table.assign(path=shortest_path)
    print(f'#final table shape: {table.shape}\n'
          f'Completed processing to build a table.')

    with open(args.output, 'w') as f:
        table.to_csv(f, sep='\t', header=True, index=False)

    elapsed_time = time.time() - start_time
    print(f'\n#time: {elasped_time} sec\n'
          f'-- fin --\n')

if __name__ == '__main__':
    main()
            
