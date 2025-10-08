import json
import os
import pandas as pd

def load_data():
    with open("data_access/data_config.json", "r") as f:
        config = json.load(f)

    base_path = config["minimal_data_path"] if config.get("use_minimal_data") else config["server_data_path"]

    with open("data_access/data_level_access.json", "r") as f:
        entry = json.load(f)

    domain = entry.get("domain")
    subdomains = entry.get("subdomain", "").split(",") if entry.get("subdomain") else []
    client = entry.get("client")

    dataframes = []

    def try_load_csv(path, label):
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(path, file))
                    dataframes.append((f"{label}:{file}", df))

    def try_load_images(images_path, label_prefix):
        if os.path.exists(images_path):
            for folder_name in os.listdir(images_path):
                folder_path = os.path.join(images_path, folder_name)
                if os.path.isdir(folder_path):
                    image_files = [
                        {"image_path": os.path.join(folder_path, file), "label": folder_name}
                        for file in os.listdir(folder_path)
                        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
                    ]
                    if image_files:
                        df_images = pd.DataFrame(image_files)
                        dataframes.append((f"{label_prefix}:{folder_name}", df_images))

    if domain:
        domain_path = os.path.join(base_path, domain)
        try_load_csv(domain_path, f"domain:{domain}")
        try_load_images(os.path.join(domain_path, "Images"), f"domain_images:{domain}")

    for sub in subdomains:
        sub_path = os.path.join(base_path, domain, sub.strip())
        try_load_csv(sub_path, f"subdomain:{sub.strip()}")
        try_load_images(os.path.join(sub_path, "Images"), f"subdomain_images:{sub.strip()}")

    if client:
        client_path = os.path.join(base_path, client)
        try_load_csv(client_path, f"client:{client}")
        try_load_images(os.path.join(client_path, "Images"), f"client_images:{client}")

    return dataframes

if __name__ == "__main__":
    dfs = load_data()
    for name, df in dfs:
        print(f"Loaded {name} with shape {df.shape}")
