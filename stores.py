import numpy as np

api_key_state = {'keys': [], 'current_index': 0, 'last_rotation': 0}

raptor_store = {
    'tree': [], 'index': None, 'filename': None,
    'levels': 0, 'leaf_nodes': []
}

excel_store = {
    'dataframes': {}, 'filename': None, 'summary': None, 'metadata': {}
}

chart_store = {'charts': [], 'counter': 0}
