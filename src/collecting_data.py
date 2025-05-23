import json 
import os
import sys 
import copy

currentdir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(currentdir, ".."))
from .processing.bigquery import bigquery_fetch_data
from .processing.batching import dynamic_batching

def fake_etl(*args, **kwargs) -> list[str]:
    """
    Store fake data from example folder into a json file
    """

    # pull_messages -> luu file json
    
    data_files = []

    for root, dirs, files in os.walk(os.path.join(currentdir, "../example")):
        for file in files:

            flag_has_file = False

            if file.endswith(".json"):
                flag_has_file = True
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    
                
            elif file.endswith(".jsonl"):
                flag_has_file = True
                with open(os.path.join(root, file), "r") as f:
                    data = []
                    for line in f:
                        data.append(json.loads(line))
                    # Process the data as needed
                    # For example, you can print it or save it to a new file

            if flag_has_file:
                if file.endswith(".json"):
                    file = file.replace(".json", ".jsonl")
                output_file = os.path.join(currentdir, "../LLaMA-Factory/data", file)
                with open(output_file, "a") as f:
                    for item in data:
                        f.write(json.dumps(item) + "\n")
                
                
                data_files.append(file)

    return data_files

    # Overwrite the dataset_info.json file


def setup_dataset_info(data_files):


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



def dynamic_batch(data_files):

    return data_files


def load_data(version):
    # Check if version already has 2 colons (format local:name:latest)
    if version.count(':') != 2:
        raise ValueError('Must have 2 :')

    print('version :', version)

    template, datasource, version = version.split(":")
    print(f"Loading data from {datasource} with template {template} and version {version}")


    data_files = []
    if datasource == "local":
        data_files = fake_etl()
    elif datasource == "bigquery":
        data_files = bigquery_fetch_data(version, template)
    

    
    data_files = dynamic_batching(data_files)
    dataset_name = setup_dataset_info(data_files)
    return dataset_name
    