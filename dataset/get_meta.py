import os
import json
from pathlib import Path

def generate_dataset_metadata():
    """
    Scans the 'data' directory to generate a 'datasets_meta.json' file
    that aggregates metadata from all found datasets with custom formatting.
    """
    try:
        # --- 1. Define Relative Paths ---
        project_root = Path.cwd()
        data_dir = project_root / 'data'
        dataset_dir = project_root / 'dataset'
        output_meta_file = dataset_dir / 'datasets_meta.json'

        # --- 2. Initialize Output Structure ---
        dataset_dir.mkdir(exist_ok=True)
        all_datasets_meta = [
            {
                "_comment": "This file aggregates metadata from all datasets in the 'data' directory. It is auto-generated.",
                "run_command": "uv run python dataset/get_meta.py"
            }
        ]
        print(f"Starting metadata generation. Output will be saved to: {output_meta_file}")

        # --- 3. Check for Data Directory ---
        if not data_dir.is_dir():
            print(f"Error: The '{data_dir}' directory was not found.")
            return

        # --- 4. Traverse Data Directory and Extract Metadata ---
        dataset_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
        for dataset_folder in dataset_folders:
            dataset_name = dataset_folder.name
            print(f"Processing dataset: {dataset_name}...")
            dataset_entry = {"Dataset": dataset_name, "Train": None, "Test": None}
            
            train_json_path = dataset_folder / 'train' / 'dataset.json'
            if train_json_path.is_file():
                with open(train_json_path, 'r') as f:
                    dataset_entry['Train'] = json.load(f)
            else:
                print(f"  - Warning: 'train/dataset.json' not found for {dataset_name}.")

            test_json_path = dataset_folder / 'test' / 'dataset.json'
            if test_json_path.is_file():
                with open(test_json_path, 'r') as f:
                    dataset_entry['Test'] = json.load(f)
            else:
                print(f"  - Warning: 'test/dataset.json' not found for {dataset_name}.")

            if dataset_entry['Train'] or dataset_entry['Test']:
                all_datasets_meta.append(dataset_entry)

        # --- 5. Write the Final JSON File with Custom, Aligned Formatting ---
        
        # First, determine the max width for each sub-field to align columns
        column_widths = {}
        all_keys = set()
        for entry in all_datasets_meta:
            if "Train" in entry and entry["Train"]:
                all_keys.update(entry["Train"].keys())  # type: ignore
            if "Test" in entry and entry["Test"]:
                all_keys.update(entry["Test"].keys())  # type: ignore
        
        # Define a canonical order for the keys
        key_order = sorted(list(all_keys))

        for key in key_order:
            max_len = 0
            for entry in all_datasets_meta:
                for sub_dict_key in ["Train", "Test"]:
                    if sub_dict_key in entry and entry[sub_dict_key] and key in entry[sub_dict_key]:
                        value = entry[sub_dict_key][key]  # type: ignore
                        # Round the specific float value before measuring
                        if key == 'mean_puzzle_examples' and isinstance(value, float):
                            value = round(value, 2)
                        
                        val_str = json.dumps(value)
                        field_str = f'"{key}": {val_str}'
                        max_len = max(max_len, len(field_str))
            column_widths[key] = max_len

        # Now, write the file using the calculated widths
        with open(output_meta_file, 'w') as f:
            f.write('[')
            for i, entry in enumerate(all_datasets_meta):
                if i > 0:
                    f.write(',\n\n')
                
                if i == 0: # Handle header entry
                    f.write(f'{{  "_comment":    "{entry["_comment"]}",\n')
                    f.write(f'    "run_command": "{entry["run_command"]}"}}')
                else: # Handle data entries
                    dataset_name = entry.get("Dataset", "")
                    f.write(f' {{  "Dataset":  "{dataset_name}",\n')
                    
                    # Train line
                    f.write(f'    "Train":    {{')
                    sub_dict = entry.get("Train")
                    if sub_dict:
                        parts = []
                        for key in key_order:
                            if key in sub_dict:
                                value = sub_dict[key]
                                # Round the specific float value for final output
                                if key == 'mean_puzzle_examples' and isinstance(value, float):
                                    value = round(value, 2)
                                field_str = f'"{key}": {json.dumps(value)}'
                                parts.append(field_str.ljust(column_widths.get(key, 0)))
                        f.write(', '.join(parts))
                    f.write('},\n')
                    
                    # Test line
                    f.write(f'    "Test":     {{')
                    sub_dict = entry.get("Test")
                    if sub_dict:
                        parts = []
                        for key in key_order:
                            if key in sub_dict:
                                value = sub_dict[key]
                                # Round the specific float value for final output
                                if key == 'mean_puzzle_examples' and isinstance(value, float):
                                    value = round(value, 2)
                                field_str = f'"{key}": {json.dumps(value)}'
                                parts.append(field_str.ljust(column_widths.get(key, 0)))
                        f.write(', '.join(parts))
                    f.write('}}')

            f.write(']')

        print("\nMetadata generation complete!")
        print(f"Successfully created or updated '{output_meta_file}'.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    generate_dataset_metadata()
