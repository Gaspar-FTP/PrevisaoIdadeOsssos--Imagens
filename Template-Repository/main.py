from data_access.load_data import load_data
import importlib.util
from pathlib import Path
import os


# Helper to load modules from the 'Code Files' folder (contains a space)
BASE = Path(__file__).parent
CODE_FILES_DIR = BASE / 'Code Files'


def load_module_from_code_files(relative_path: str, module_name: str):
    module_path = CODE_FILES_DIR.joinpath(relative_path)
    if not module_path.exists():
        raise FileNotFoundError(f"Module file not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the modular implementations
dp_mod = load_module_from_code_files('data_processing_scripts\\data_preprocessing.py', 'data_preprocessing')
train_mod = load_module_from_code_files('training_scripts\\training_scripts.py', 'training_scripts')
eval_mod = load_module_from_code_files('evaluation_scripts\\evaluation_scripts.py', 'evaluation_scripts')


def preprocess_data(dataframes):
    return dp_mod.preprocess_data(dataframes)


def train_models(processed_data):
    return train_mod.train_models(processed_data, logs_dir=os.path.join(os.getcwd(), 'logs'))


def evaluate_models(model, processed_data):
    return eval_mod.evaluate_and_save(model, processed_data, processed_data['std_bone_age'], processed_data['mean_bone_age'])


def main():
    dataframes = load_data()
    # debug: inspecionar 'dataframes' retornado por load_data()
    print("DEBUG: tipo(dataframes) =", type(dataframes))
    # se for iterável, imprime até 10 items com tipo e (se for DataFrame) suas colunas
    try:
        it = iter(dataframes)
        for i, item in enumerate(dataframes):
            if i >= 10:
                break
            print(f"  item[{i}] type:", type(item))
            # se for tuple (name, df) ou similar
            if isinstance(item, tuple) and len(item) == 2:
                print("    tuple[0] type:", type(item[0]), "  tuple[1] type:", type(item[1]))
                if hasattr(item[1], "columns"):
                    print("    columns:", list(item[1].columns)[:10])
            elif hasattr(item, "columns"):
                print("    DataFrame columns:", list(item.columns)[:10])
            elif isinstance(item, str):
                print("    string path:", item)
    except Exception as e:
        print("DEBUG: não foi possível iterar dataframes:", e)

    print("\nLoaded datasets:")
    for name, df in dataframes:
        print(f"- {name}: shape {df.shape}")
        print(df.head(2))

    processed_data = preprocess_data(dataframes)
    model, history = train_models(processed_data)
    evaluate_models(model, processed_data)


if __name__ == "__main__":
    main()
