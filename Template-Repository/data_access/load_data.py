import json
import os

def load_data():
    # Load data_config.json to determine base path
    with open("data_config.json", "r") as f:
        config = json.load(f)

    base_path = config["minimal_data_path"] if config.get("use_minimal_data") else config["server_data_path"]
    print(f"Base data path: {base_path}")

    # Load data_level_access.json and parse the single entry
    with open("data_level_access.json", "r") as f:
        entry = json.load(f)

    domain = entry.get("domain")
    subdomain = entry.get("subdomain")
    client = entry.get("client")

    # Print applicable paths
    if domain:
        print(os.path.join(base_path, domain))
    if domain and subdomain:
        for sub in subdomain.split(","):
            print(os.path.join(base_path, domain, sub.strip()))
    if client:
        print(os.path.join(base_path, client))

if __name__ == "__main__":
    load_data()
