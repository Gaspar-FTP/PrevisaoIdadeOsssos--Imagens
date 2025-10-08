# training_scripts.py
import os
import pickle
import datetime
import tempfile
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import GlobalMaxPooling2D, Flatten, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

_PROCESSED_PICKLE = "processed_data.pkl"
_CHECKPOINT = "best_model.h5"
_HISTORY_PICKLE = "train_history.pkl"

try:
    from tqdm.keras import TqdmCallback
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


def _tf_mae_in_months(std_bone_age, mean_bone_age):
    def mae_in_months(y_true, y_pred):
        # Ensure std and mean are cast to a common tensor dtype to avoid float32/float64 mismatches.
        # Use y_pred.dtype as the canonical dtype (y_pred may come from model outputs and can differ
        # from y_true). Cast y_true and y_pred to that dtype as well.
        try:
            target_dtype = y_pred.dtype
        except Exception:
            # Fallback: if y_pred has no dtype attribute for some reason, use y_true's dtype
            target_dtype = y_true.dtype

        std_t = tf.cast(std_bone_age, target_dtype)
        mean_t = tf.cast(mean_bone_age, target_dtype)

        y_true_cast = tf.cast(y_true, target_dtype)
        y_pred_cast = tf.cast(y_pred, target_dtype)

        true_months = y_true_cast * std_t + mean_t
        pred_months = y_pred_cast * std_t + mean_t
        return tf.reduce_mean(tf.abs(true_months - pred_months))
    return mae_in_months


def train_models(processed_data, logs_dir: str = None):
    """
    Recebe processed_data (o dicionário retornado por preprocess_data) e treina o modelo.
    Também suporta ser chamado com processed_data==None, caso em que tenta ler processed_data.pkl.
    Grava best_model.h5 e train_history.pkl.
    """
    # se processed_data for None, carrega do ficheiro
    if processed_data is None:
        if not os.path.exists(_PROCESSED_PICKLE):
            raise FileNotFoundError(f"processed_data is None and {_PROCESSED_PICKLE} not found.")
        with open(_PROCESSED_PICKLE, "rb") as f:
            processed_data = pickle.load(f)

    train_gen = processed_data.get("train_generator")
    val_gen = processed_data.get("val_generator")
    mean_bone_age = processed_data.get("mean_bone_age")
    std_bone_age = processed_data.get("std_bone_age")
    img_size = processed_data.get("img_size", 256)

    # GPU configuration: enable memory growth to avoid TF allocating all GPU memory
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for g in gpus:
                try:
                    tf.config.experimental.set_memory_growth(g, True)
                except Exception:
                    # set_memory_growth may not be available on some builds; ignore if it fails
                    pass
            print(f"Detected GPU(s): {[getattr(g, 'name', str(g)) for g in gpus]}")
        else:
            print("No GPU detected; training will run on CPU.")
    except Exception as e:
        print(f"Warning: unable to query/initialize GPUs: {e}")

    # Optional: enable mixed precision via environment variable TF_MIXED_PRECISION=1
    try:
        if os.environ.get('TF_MIXED_PRECISION', '0') in ('1', 'true', 'True'):
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            print('Mixed precision enabled: policy set to mixed_float16')
            print('Note: the final Dense layer is forced to float32 to avoid numeric issues.')
    except Exception as e:
        print(f"Warning: could not enable mixed precision: {e}")

    # build model (Xception base, por defeito congelado para treino rápido)
    base = Xception(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
    base.trainable = False

    model = Sequential([
        base,
        GlobalMaxPooling2D(),
        Flatten(),
        Dense(10, activation='relu'),
        # ensure final output is float32 even if mixed precision is active
        Dense(1, activation='linear', dtype='float32')
    ])

    model.compile(loss='mse', optimizer='adam', metrics=[_tf_mae_in_months(std_bone_age, mean_bone_age)])
    model.summary()

    # callbacks
    if logs_dir is None:
        logs_dir = "logs"
    # If a file exists with the same name, rename it to a backup and create the directory
    if os.path.exists(logs_dir) and not os.path.isdir(logs_dir):
        ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        backup_name = f"{logs_dir}.bak.{ts}"
        try:
            os.rename(logs_dir, backup_name)
            print(f"Renamed existing file '{logs_dir}' to backup '{backup_name}' to create logs directory.")
        except Exception as e:
            raise RuntimeError(f"Cannot create logs directory because a non-directory exists at '{logs_dir}' and renaming failed: {e}")
    os.makedirs(logs_dir, exist_ok=True)
    # Prefer to write TensorBoard logs to a temp dir when path contains non-ASCII or creation fails
    tb = None
    try:
        candidate = os.path.join(logs_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        # detect non-ascii in path (Windows TF sometimes fails with non-ascii usernames/OneDrive)
        if any(ord(ch) > 127 for ch in candidate):
            tmp_tb = tempfile.mkdtemp(prefix='tf_logs_')
            print(f"Notice: project path contains non-ASCII characters, using temporary TB dir: {tmp_tb}")
            tb = TensorBoard(log_dir=tmp_tb)
        else:
            try:
                os.makedirs(candidate, exist_ok=True)
                tb = TensorBoard(log_dir=candidate)
            except Exception:
                tmp_tb = tempfile.mkdtemp(prefix='tf_logs_')
                print(f"Notice: could not use {candidate} for TB logs, using temporary dir {tmp_tb}")
                tb = TensorBoard(log_dir=tmp_tb)
    except Exception as e:
        print(f"Warning: could not create TensorBoard callback: {e}. Continuing without TensorBoard.")
    mc = ModelCheckpoint(_CHECKPOINT, monitor='val_loss', mode='min', save_best_only=True)
    es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

    callbacks = [tb, mc, es, rlr]
    # remove None callbacks (if tb creation failed)
    callbacks = [c for c in callbacks if c is not None]
    if _HAS_TQDM:
        callbacks.append(TqdmCallback(verbose=1))

    # Diagnostic: inspect one batch from generators (shapes)
    try:
        if train_gen is not None:
            batch = next(iter(train_gen))
            if isinstance(batch, tuple) and len(batch) >= 1:
                x = batch[0]
                print(f"Debug: train batch x shape: {getattr(x, 'shape', type(x))}")
            # reset generator if supported
            try:
                train_gen.reset()
            except Exception:
                pass
        if val_gen is not None:
            batchv = next(iter(val_gen))
            if isinstance(batchv, tuple) and len(batchv) >= 1:
                xv = batchv[0]
                print(f"Debug: val batch x shape: {getattr(xv, 'shape', type(xv))}")
            try:
                val_gen.reset()
            except Exception:
                pass
    except Exception as e:
        print(f"Warning: could not sample batch for debug: {e}")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=1,
        callbacks=callbacks,
        verbose=0 if _HAS_TQDM else 1
    )

    # gravar history
    with open(_HISTORY_PICKLE, "wb") as f:
        pickle.dump(history.history, f)

    # garantir que o melhor modelo está salvo (ModelCheckpoint faz isto, mas garantir)
    model.save(_CHECKPOINT)

    print(f"Training finished. Model saved to {_CHECKPOINT} and history to {_HISTORY_PICKLE}.")

    # retornar model e history para permitir avaliação imediata
    return model, history
