import copy
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def reorganize_dict(t):
    # Create a new dictionary to store the reorganized data
    new_dict = {}

    # Loop through the outer dictionary (models)
    for model, datasets in t.items():
        # Loop through the inner dictionary (datasets)
        for dataset, value in datasets.items():
            # If the dataset doesn't exist in the new dict, create an entry for it
            if dataset not in new_dict:
                new_dict[dataset] = {}
            # Assign the model and its corresponding value to the new dict under the dataset
            new_dict[dataset][model] = copy.deepcopy(value)

    return new_dict

