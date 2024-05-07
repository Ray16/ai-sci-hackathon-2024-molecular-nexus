import json

train_data = json.load(open('data.json'))
example_smiles = list(train_data.keys())[0]
example_property = list(train_data.values())[0]
example_data = {example_smiles:example_property}
with open("example_data.json", "w") as f: 
    json.dump(example_data, f)