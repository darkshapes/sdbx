import json
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description="Compare model state dicts for differences")

parser.add_argument('file1', type=str, help='First file to compare')
parser.add_argument('file2', type=str, help='Second file to compare')
args = parser.parse_args()

if Path(args.file1).exists():
    path_location = Path(args.file1)
    with open(args.file1, "r") as file:
        state_dict_1 = json.load(file)
else:
    print(f"The file {args.file1} does not exist.")

if Path(args.file2).exists():
    path_location = Path(args.file2)
    with open(path_location, "r") as file:
        state_dict_2 = json.load(file)
else:
    print(f"The file {args.file1} does not exist.")


def compare_dicts(dict1, dict2):
    # Convert dict keys to sets for quick comparison
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    # Check differences in keys
    keys_only_in_dict1 = keys1 - keys2
    keys_only_in_dict2 = keys2 - keys1
    common_keys = keys1 & keys2

    # Check differences in values for common keys
    value_differences = {k: (dict1[k], dict2[k]) for k in common_keys if k == "shape" and dict1[k] != dict2[k]}

    return {
        "keys_only_in_A": keys_only_in_dict1,
        "keys_only_in_B": keys_only_in_dict2,
        "value_differences": value_differences,
    }



if state_dict_1 is not None and state_dict_2 is not None:
    result = compare_dicts(state_dict_1, state_dict_2)
    if state_dict_1 is not None and state_dict_2 is not None:
        filename = os.path.basename(Path(args.file1)) + os.path.basename(Path(args.file2))
        result = compare_dicts(state_dict_1, state_dict_2)
        print(result)
        with open(filename, "w", encoding="UTF-8") as comparison: # todo: make 'a' type before release
            json.dump(str(result), comparison, ensure_ascii=False, indent=4, sort_keys=False)

