import networkx as nx
import helper

train_data = helper.load_data_from_file("data.json")

for smiles_string in train_data:
    graph = train_data[smiles_string]
    for idx,node in enumerate(graph.nodes(data=True)):
        print(node)
        if idx % 11 == 0:
            break

    for idx, edge in enumerate(graph.edges(data=True)):
        print(edge)
        if idx % 11 == 0:
            break