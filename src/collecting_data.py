import json 
import os
import sys 
import copy

currentdir = os.path.dirname(os.path.abspath(__file__))


def fake_etl(*args, **kwargs) -> list[str]:
    """
    Store fake data from example folder into a json file
    """

    # pull_messages -> luu file json
    
    data_files = []

    for root, dirs, files in os.walk(os.path.join(currentdir, "../example")):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    # Process the data as needed
                    # For example, you can print it or save it to a new file
                
                # Save to a new file
                output_file = os.path.join(currentdir, "../LLaMA-Factory/data", file)
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=4)
                data_files.append(file)

    # Overwrite the dataset_info.json file

    sample_metadata = {
        "file_name": "sample.json",
        "formatting": "sharegpt",
        "columns": {
        "messages": "messages"
        },
        "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system"
        }
    }

    dataset_name = []
    with open(os.path.join(currentdir, "../LLaMA-Factory/data/dataset_info.json"), "r") as f:
        dataset_info = json.load(f)
    
    
    for file in data_files:
        base_name = os.path.basename(file).replace(".json", "")

        config = copy.deepcopy(sample_metadata)
        config["file_name"] = file
        dataset_name.append(base_name)
        dataset_info[base_name] = config

    with open(os.path.join(currentdir, "../LLaMA-Factory/data/dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=4)

    return dataset_name    