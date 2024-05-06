import json
from helper import load_data_from_file

data = json.load(open('data.json'))
print(f'Number of molecules: {len(data)}')

# way to access data:
# first convert keys and values into list
# and then index

# load graph
graphs = load_data_from_file('data.json')
print(type(graphs))
print(list(graphs.keys())[0])
print(list(graphs.values())[0])