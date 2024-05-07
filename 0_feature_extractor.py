import torch
from mace_layer import MACE_layer
from helper import load_data_from_file

train_data = load_data_from_file('data.json')

for smiles_string in train_data:
    graph = train_data[smiles_string]
    node_labels = []
    node_feats = [] # dim: num_nodes x node_feats_irreps
    node_attrs = [] # dim: num_nodes x n_dims_in
    # append node features and attributes
    for node in graph.nodes(data=True):
        print(f'node: {node}')
        # append node label
        node_labels.append(node[0])
        # treat the following as node attributes
        node_attrs_single_node = []
        for attr in ['atomic', 'valence', 'formal_charge', 'aromatic', 'hybridization', "radical_electrons"]:
            #print(attr, node[1][attr])
            node_attrs_single_node.append(node[1][attr])
        node_attrs.append(node_attrs_single_node)

        # treat the following as prediction target
        # need to collect information
        for key in ['mass', 'charge', 'sigma', 'epsilon']:
            print(key, node[1]["param"][key])
    
    vectors = [] # dim: num_edges x 3
    edge_feats = [] # dim: num_edges x egde_feats_irreps
    edge_index = [] # dim: 2 x num_edges
    # append edge features and attributes
    for edge in graph.edges(data=True):
        edge_index.append(edge[0],edge[1])
        
        edge_prop_single_edge = []
        for edge_prop in ['type', 'stereo', 'aromatic', 'conjugated']:
            #print(edge_prop, edge[2][edge_prop])
            edge_prop_single_edge.append(edge[2][edge_prop])
        edge_feats.append(edge_prop_single_edge)
    
    # convert all inputs into tensor form
    vectors = torch.tensor(vectors)
    node_feats = torch.tensor(node_feats)
    node_attrs = torch.tensor(node_attrs)
    edge_feats = torch.tensor(edge_feats)
    edge_index = torch.tensor(edge_index)
    # transpose edge_index to match mace input dimensions
    edge_index = torch.transpose(edge_index)

    # pass all informaiton into MACE_layer
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
    print(node_feats)
    break