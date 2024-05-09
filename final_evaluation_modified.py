import networkx as nx
import numpy as np
import helper as helper
import hashlib
import copy
import json

from make_permutation import get_inv_permutation, apply_permutation

def compare_property(property_name:str, result_dict, ref_dict, max_node_size:int=100):
    def get_graph_property_data(property_name, graph):
        data = np.zeros(max_node_size)
        for node in graph.nodes(data=True):
            data[node[0]] = node[1]["param"][property_name]
        return data
    result_data = []
    ref_data = []

    miss_counter = 0
    for smi in ref_dict:
        m = hashlib.shake_256()
        m.update(bytes(smi, "utf-8"))
        name = m.hexdigest(10)
        if name not in result_dict:
            miss_counter += 1
        else:
            result_data.append(get_graph_property_data(property_name, result_dict[name]))
            ref_data.append(get_graph_property_data(property_name, ref_dict[smi]))
    result_data = np.asarray(result_data)
    ref_data = np.asarray(ref_data)

    negative_counter = np.sum(ref_data < 0)

    print(f"\nAnalysis for {property_name}")
    print(f"# Missing Molecules: {miss_counter}")
    print(f"# Negative values detected: {negative_counter}")

    sq_diff = (result_data - ref_data)**2
    print(f"Root Mean Squared Difference: {np.sqrt(np.mean(sq_diff))}")
    print(f"Max difference: {np.sqrt(np.max(sq_diff))}")
    print("\n")


# the function below is modified
def add_data_from_prediction(result_dict, output_filename, predictions):
    def get_ff_param(pred_node):
        param = {}
        param["mass"] = pred_node[0].numpy().item()
        param["charge"] = pred_node[1].numpy().item()
        param["sigma"] = pred_node[2].numpy().item()
        param["epsilon"] = pred_node[3].numpy().item()
        return param
    
    for mol_idx, name in enumerate(result_dict):
        graph = result_dict[name]
        for node_idx, node in enumerate(graph.nodes(data=True)):
            pred_node = predictions[mol_idx][node_idx]
            node[1]["param"] = get_ff_param(pred_node)
            graph.update(nodes=[node])

        result_dict[name] = graph
        # result_dict
    helper.write_data_to_json_file(result_dict, output_filename, indent=2)

    
    
def compare_permutation(property_name:str, result_dict, ref_graph, permutation_dict, max_node_size:int=100):
    def get_graph_property_data(property_name, graph):
        data = np.zeros(max_node_size)
        for node in graph.nodes(data=True):
            data[node[0]] = node[1]["param"][property_name]
        return data
    result_data = []
    ref_data = []

    miss_counter = 0
    for name in permutation_dict:
        perm = np.asarray(permutation_dict[name], dtype=int)
        if name not in result_dict:
            miss_counter += 1
        else:
            inv_permutation = get_inv_permutation(perm)
            graph = result_dict[name]
            inv_graph = apply_permutation(graph, inv_permutation)

            # Check that inverting the permutation worked
            for node in ref_graph.nodes(data=True):
                ref_attr = copy.deepcopy(node[1])
                del ref_attr["param"]
                result_attr = copy.deepcopy(inv_graph.nodes(data=True)[node[0]])
                del result_attr["param"]

                assert ref_attr == result_attr

            result_data.append(get_graph_property_data(property_name, inv_graph))
            ref_data.append(get_graph_property_data(property_name, ref_graph))
    result_data = np.asarray(result_data)
    ref_data = np.asarray(ref_data)

    negative_counter = np.sum(ref_data < 0)

    print(f"\nAnalysis for {property_name}")
    print(f"# Missing Molecules: {miss_counter}")
    print(f"# Negative values detected: {negative_counter}")

    sq_diff = (result_data - ref_data)**2
    print(f"Root Mean Squared Difference: {np.sqrt(np.mean(sq_diff))}")
    print(f"Max difference: {np.sqrt(np.max(sq_diff))}")
    print("\n")




def main():
    # You won't necessarily have this data available, here we use the training data to show you
    ref_dict = helper.load_data_from_file("data.json")


    result_dict = helper.load_data_from_file("validation_example.json")

    # In a real case, we would not add random results.
    # Instead you would fill it with your results and then write it to disk to hand it to us.
    # This is for mock up testing only.
    rng = np.random.default_rng()
    add_data_from_prediction(result_dict, rng)

     
    compare_property("epsilon", result_dict, ref_dict)
    compare_property("mass", result_dict, ref_dict)
    compare_property("sigma", result_dict, ref_dict)
    compare_property("charge", result_dict, ref_dict)


    print("Permutation check")
    # The real SMILES string we will test with is not in your training data
    ref_graph = ref_dict["O=C(c1ccc2c(c1)OCO2)c1ccc2n1CCC2C(=O)O"]
    result_perm_dict = helper.load_data_from_file("permutation_example_masked.json")
    with open("permutation_example.json", "r") as json_handle:
        permutation_dict = json.load(json_handle)
    # This step is again replaced with your model data
    add_data_from_prediction(result_perm_dict, rng)

    compare_permutation("epsilon", result_perm_dict, ref_graph, permutation_dict)
    compare_permutation("mass", result_perm_dict, ref_graph, permutation_dict)
    compare_permutation("sigma", result_perm_dict, ref_graph, permutation_dict)
    compare_permutation("charge", result_perm_dict, ref_graph, permutation_dict)




if __name__ == "__main__":
    main()
