# evaluation_scripts.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

_PROCESSED_PICKLE = "processed_data.pkl"
_CHECKPOINT = "best_model.h5"
_RESULTS_CSV = "results.csv"
_PLOT_PATH = "pred_vs_actual.png"


def evaluate_models():
    """
    Carrega processed_data.pkl e best_model.h5 do disco, faz previsões no test generator (se houver),
    grava results.csv e um gráfico pred_vs_actual.png se possível.
    Assinatura preservada: evaluate_models() sem argumentos.
    """
    if not os.path.exists(_PROCESSED_PICKLE):
        raise FileNotFoundError(f"{_PROCESSED_PICKLE} not found. Execute preprocess_data first.")

    with open(_PROCESSED_PICKLE, "rb") as f:
        processed_data = pickle.load(f)

    test_gen = processed_data.get("test_generator")
    val_gen = processed_data.get("val_generator")
    mean_bone_age = processed_data.get("mean_bone_age")
    std_bone_age = processed_data.get("std_bone_age")

    if not os.path.exists(_CHECKPOINT):
        raise FileNotFoundError(f"{_CHECKPOINT} not found. Execute train_models first.")

    # carrega modelo (usa custom_objects se necessário; a métrica é pura TF e deve ser carregável)
    model = load_model(_CHECKPOINT, compile=False)

    if test_gen is None:
        print("No test generator available. Skipping predictions.")
        return

    preds = model.predict(test_gen, verbose=1)
    preds = np.asarray(preds).ravel()
    predicted_months = mean_bone_age + std_bone_age * preds
    filenames = test_gen.filenames

    results = pd.DataFrame({"Filename": filenames, "Prediction_months": predicted_months})
    results.to_csv(_RESULTS_CSV, index=False)
    print(f"Saved predictions to {_RESULTS_CSV}")

    # Se tivermos val_gen com labels, podemos fazer um gráfico actual vs pred (usa val_gen como proxy)
    if val_gen is not None:
        # recolhe y_true do val_gen (cuidado com memória; usa somente até len(preds) se necessário)
        y_trues = []
        for i in range(len(val_gen)):
            bx, by = val_gen[i]
            y_trues.append(by)
        y_trues = np.concatenate(y_trues).ravel()
        true_months = mean_bone_age + std_bone_age * y_trues
        # truncar ao min
        n = min(len(true_months), len(predicted_months))
        if n > 0:
            plt.figure(figsize=(7,7))
            plt.plot(true_months[:n], predicted_months[:n], 'r.', label='predictions')
            mn, mx = min(true_months[:n].min(), predicted_months[:n].min()), max(true_months[:n].max(), predicted_months[:n].max())
            plt.plot([mn, mx], [mn, mx], 'b-', label='actual')
            plt.xlabel('Actual Age (Months)')
            plt.ylabel('Predicted Age (Months)')
            plt.legend()
            plt.savefig(_PLOT_PATH, dpi=200)
            plt.close()
            print(f"Saved plot to {_PLOT_PATH}")


def evaluate_and_save(model, processed_data, std_bone_age, mean_bone_age, out_csv: str = _RESULTS_CSV):
    """
    Evaluate provided `model` on `processed_data` and save predictions to CSV.
    Signature matches the call from `main.py`.
    """
    # processed_data is expected to be the dict returned by preprocess_data
    test_gen = processed_data.get("test_generator")
    val_gen = processed_data.get("val_generator")

    if test_gen is None:
        print("No test generator available in processed_data. Skipping predictions.")
        return

    # If a path string was passed instead of a model, try to load it
    try:
        # Keras Model objects have a predict method; if not, try to load
        has_predict = callable(getattr(model, "predict", None))
    except Exception:
        has_predict = False

    if not has_predict and isinstance(model, str) and os.path.exists(model):
        model = load_model(model, compile=False)

    preds = model.predict(test_gen, verbose=1)
    preds = np.asarray(preds).ravel()

    # Compute months from normalized predictions
    predicted_months = mean_bone_age + std_bone_age * preds

    # Try to get filenames from generator; fallback to index-based names
    filenames = getattr(test_gen, 'filenames', None)
    if filenames is None:
        try:
            filenames = [f"sample_{i}" for i in range(len(predicted_months))]
        except Exception:
            filenames = list(range(len(predicted_months)))

    results = pd.DataFrame({"Filename": filenames, "Prediction_months": predicted_months})
    results.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")

    # Optionally create a pred vs actual plot using val_gen as proxy (if available)
    if val_gen is not None:
        y_trues = []
        for i in range(len(val_gen)):
            bx, by = val_gen[i]
            y_trues.append(by)
        y_trues = np.concatenate(y_trues).ravel()
        true_months = mean_bone_age + std_bone_age * y_trues
        n = min(len(true_months), len(predicted_months))
        if n > 0:
            plt.figure(figsize=(7,7))
            plt.plot(true_months[:n], predicted_months[:n], 'r.', label='predictions')
            mn = min(true_months[:n].min(), predicted_months[:n].min())
            mx = max(true_months[:n].max(), predicted_months[:n].max())
            plt.plot([mn, mx], [mn, mx], 'b-', label='actual')
            plt.xlabel('Actual Age (Months)')
            plt.ylabel('Predicted Age (Months)')
            plt.legend()
            plt.savefig(_PLOT_PATH, dpi=200)
            plt.close()
            print(f"Saved plot to {_PLOT_PATH}")
