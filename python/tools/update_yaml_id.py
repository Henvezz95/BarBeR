import argparse
import re
from ruamel.yaml import YAML

def update_ids_in_strings(obj, old_id, new_id):
    """
    Recursively update strings in a Python object (parsed from YAML).
    Replaces '_old_id.<extension>' with '_new_id.<extension>' ONLY if 
    '_old_id' is immediately before the file extension at the end of the string.
    For example:
      "model_0.pt" -> "model_1.pt"
    but does NOT replace "model_0.25x_foo.pt".
    """
    # Compile a pattern that matches: _old_id.<anything that isn't '.'> until end of string
    # e.g., _0.pt or _0.onnx or _0.whatever
    pattern = re.compile(rf"_{old_id}(\.[^.]+)$")

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                update_ids_in_strings(value, old_id, new_id)
            elif isinstance(value, str):
                # Replace only if '_old_id' is right before the extension at the end of the string
                obj[key] = pattern.sub(f"_{new_id}\\1", value)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            if isinstance(item, (dict, list)):
                update_ids_in_strings(item, old_id, new_id)
            elif isinstance(item, str):
                obj[index] = pattern.sub(f"_{new_id}\\1", item)

def replace_id_in_yaml(input_path, output_path, old_id, new_id):
    """
    Reads a YAML file, replaces occurrences of '_old_id.<extension>' with '_new_id.<extension>'
    ONLY if it appears at the end of the string, then writes the modified YAML 
    to `output_path`, preserving aliases/comments via ruamel.yaml.
    """
    yaml = YAML()
    with open(input_path, 'r', encoding='utf-8') as f:
        data = yaml.load(f)

    # Recursively update all string fields in the YAML data structure
    update_ids_in_strings(data, old_id, new_id)

    # Save the updated data back to a YAML file
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)

def main():
    parser = argparse.ArgumentParser(
        description="Replace '_old_id.<ext>' with '_new_id.<ext>' at the end of filenames in a YAML."
    )
    parser.add_argument("input_path", help="Path to the input YAML file")
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="Path to the output YAML file (if omitted, the input file is overwritten)."
    )
    parser.add_argument("--old_id", required=True, help="Old ID to replace (integer in the filename).")
    parser.add_argument("--new_id", required=True, help="New ID to replace with.")
    args = parser.parse_args()

    # Determine whether to overwrite the original file
    output_path = args.input_path if args.output_path is None else args.output_path
    replace_id_in_yaml(args.input_path, output_path, args.old_id, args.new_id)

if __name__ == "__main__":
    main()
