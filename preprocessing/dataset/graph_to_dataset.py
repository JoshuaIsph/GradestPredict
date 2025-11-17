# preprocessing/dataset/graph_to_dataset.py

def paths_to_dataset(paths, target_node, source=None):
    """
    Convert multiple paths into supervised dataset entries.
    Adds an optional 'source' column to indicate which pathfinding algorithm generated it.
    """
    dataset = []
    for path in paths:
        for i in range(len(path) - 1):
            row = {
                'current_hold': path[i],
                'target_hold': target_node,
                'next_hold': path[i + 1]
            }
            if source:
                row['source'] = source  # add source column
            dataset.append(row)
    return dataset
