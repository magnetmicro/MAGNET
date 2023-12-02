import pandas as pd
import numpy as np
import ast
import csv


def get_class_nb(system_name):
    classList = set()
    with open(system_name+'/classesList.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file,)
        for row in csv_reader:
            classList.add(row[0])
    return len(classList)
    
def export_to_ndz(system_name, feature_file, call_graph_file):
    # Load node features
    features_df = pd.read_csv(feature_file, delimiter=";",header=None, names=['node_name', 'features'])

    # Load method call relationships
    edges_df = pd.read_csv(call_graph_file, delimiter=";",header=None, names=['node1', 'node2'])
    all_nodes = sorted(set(features_df['node_name']) | set(edges_df['node1']) | set(edges_df['node2']))
    node_to_index = {node: idx for idx, node in enumerate(all_nodes)}
    num_nodes = len(node_to_index)
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    for _, row in edges_df.iterrows():
        i = node_to_index[row['node1']]
        j = node_to_index[row['node2']]
        adjacency_matrix[i][j] = 1
        adjacency_matrix[j][i] = 1  # If the graph is undirected
    # Initialize a matrix of zeros
    num_nodes = len(node_to_index)
    num_features = len(ast.literal_eval(features_df.iloc[0]['features']))
    feature_matrix = np.zeros((num_nodes, num_features))
    
    # Fill the matrix with the features in the correct order
    for _, row in features_df.iterrows():
        node_idx = node_to_index[row['node_name']]
        try:
            features = np.array(ast.literal_eval(row['features']))
        except ValueError:
            features = np.zeros(100)
            print(f"array of zero {features}")
        feature_matrix[node_idx] = features
    np.savez(system_name+'/graph_data.npz', adjacency=adjacency_matrix, features=feature_matrix)
    return all_nodes