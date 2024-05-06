import json
from mace_layer import MACE_layer
from helper import load_data_from_file

graphs = load_data_from_file('data.json')
print(type(graphs))
print(list(graphs.keys())[0])
print(list(graphs.values())[0])

'''
layer = MACE_layer(
    max_ell=3,
    correlation=3,
    n_dims_in=2,
    hidden_irreps="16x0e + 16x1o + 16x2e",
    node_feats_irreps="16x0e + 16x1o + 16x2e",
    edge_feats_irreps="16x0e",
    avg_num_neighbors=10.0,
    use_sc=True,
)
node_feats = layer(
    vectors,
    node_feats,
    node_attrs,
    edge_feats,
    edge_index,
)

'''