# substitui/cola isto no Code Files/data_processing_scripts/data_preprocessing.py
import os
import json
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input

_PROCESSED_PICKLE = "processed_data.pkl"


def _resolve_base_path_from_config():
    cfg_path = os.path.join("data_access", "data_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"data_config.json not found at {cfg_path}")
    with open(cfg_path, "r", encoding="utf8") as f:
        cfg = json.load(f)
    base_path = cfg.get("minimal_data_path") if cfg.get("use_minimal_data") else cfg.get("server_data_path")
    return base_path


def _guess_image_dirs(base_path):
    """
    Procura as pastas de imagens de treino/test.
    Trata o caso em que existe uma pasta com o nome padrão que contém
    uma subpasta com o mesmo nome onde estão efectivamente as imagens,
    e também procura one-level-deep pastas com muitas imagens.
    Retorna (train_img_dir, test_img_dir) ou (None, None).
    """
    candidates = {
        "train": ["boneage-training-dataset", "boneage-training", "training", "train"],
        "test": ["boneage-test-dataset", "boneage-test", "test"]
    }
    found = {}

    for key, names in candidates.items():
        for name in names:
            p = os.path.join(base_path, name)
            if os.path.isdir(p):
                # se p existe, verifica se as imagens estão diretamente em p
                imgs = [f for f in os.listdir(p) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                if len(imgs) > 0:
                    found[key] = p
                    break
                # se p só contém uma subpasta com mesmo (ou qualquer) nome, e essa subpasta tem imagens, entra nela
                subdirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
                # Preferir subdir com mesmo nome
                same_name_sub = os.path.join(p, os.path.basename(p))
                if os.path.isdir(same_name_sub):
                    imgs2 = [f for f in os.listdir(same_name_sub) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                    if len(imgs2) > 0:
                        found[key] = same_name_sub
                        break
                # caso contrário, procura qualquer subdir que contenha imagens
                for sd in subdirs:
                    sd_full = os.path.join(p, sd)
                    imgs_sd = [f for f in os.listdir(sd_full) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                    if len(imgs_sd) > 0:
                        found[key] = sd_full
                        break
                if key in found:
                    break

    # fallback: detect directories with many images (one level deep)
    if "train" not in found or "test" not in found:
        for root, dirs, files in os.walk(base_path):
            imgs = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if len(imgs) >= 50 and "train" not in found:
                found["train"] = root
            elif len(imgs) >= 10 and "test" not in found:
                found["test"] = root
            if "train" in found and "test" in found:
                break

    return found.get("train"), found.get("test")



def _is_filename_column(colname):
    col = colname.lower().strip()
    return col in ("id", "filename", "file", "image", "case id", "case_id", "name")


def _is_label_column(colname):
    col = colname.lower().strip()
    return col in ("boneage", "bone_age", "age", "target")


def _find_training_csv_in_dataframes(dataframes):
    """
    Retorna um tuple (train_df, maybe_labels_df) onde:
      - train_df: DataFrame que contém filenames (col id/filename) OR já tem boneage
      - maybe_labels_df: None ou DataFrame separado com labels que podemos mergear
    Aceita vários formatos em 'dataframes' (lista de tuples, list of df, dict, ...).
    """
    import pandas as _pd

    # normalize input to list
    items = []
    if isinstance(dataframes, dict):
        items = list(dataframes.items())
    else:
        try:
            items = list(dataframes)
        except Exception:
            items = [dataframes]

    train_df = None
    labels_df = None

    for it in items:
        # tuple (name, df)
        if isinstance(it, tuple) and len(it) == 2:
            name, cand = it
            # if cand is DataFrame
            if hasattr(cand, "columns"):
                cols = [c.lower() for c in cand.columns]
                # contains both filenames and labels => this is ideal
                if any(_is_filename_column(c) for c in cols) and any(_is_label_column(c) for c in cols):
                    return cand.copy(), None
                # contains only labels (age) but no filenames -> treat as labels_df
                if any(_is_label_column(c) for c in cols) and not any(_is_filename_column(c) for c in cols):
                    labels_df = cand.copy()
                    continue
                # contains filenames but not labels -> candidate train_df
                if any(_is_filename_column(c) for c in cols):
                    train_df = cand.copy()
                    continue
        else:
            cand = it
            # if cand is DataFrame
            if hasattr(cand, "columns"):
                cols = [c.lower() for c in cand.columns]
                if any(_is_filename_column(c) for c in cols) and any(_is_label_column(c) for c in cols):
                    return cand.copy(), None
                if any(_is_label_column(c) for c in cols) and not any(_is_filename_column(c) for c in cols):
                    labels_df = cand.copy()
                    continue
                if any(_is_filename_column(c) for c in cols):
                    train_df = cand.copy()
                    continue
            # if cand is a path string -> try read
            if isinstance(cand, str):
                try:
                    df = _pd.read_csv(cand)
                    cols = [c.lower() for c in df.columns]
                    if any(_is_filename_column(c) for c in cols) and any(_is_label_column(c) for c in cols):
                        return df.copy(), None
                    if any(_is_filename_column(c) for c in cols):
                        if train_df is None:
                            train_df = df.copy()
                    if any(_is_label_column(c) for c in cols):
                        if labels_df is None:
                            labels_df = df.copy()
                except Exception:
                    pass

    return train_df, labels_df


def preprocess_data(dataframes):
    """
    Versão robusta: tenta extrair train CSV a partir de `dataframes` (o que load_data() devolve).
    Se não encontrar, faz fallback e procura ficheiros em base_path (data_access config).
    Trata casos:
      - CSV com filenames + labels
      - CSV só com filenames (cria generator sem labels)
      - ficheiros CSV ou Excel (.csv, .xlsx, .xls)
    No final devolve 'processed_data' (dict) e grava processed_data.pkl.
    """
    import pandas as _pd
    # 1) tenta obter train_df a partir dos dataframes passados
    train_df, labels_df = _find_training_csv_in_dataframes(dataframes)

    # 2) se não encontrado, procura ficheiros diretamente no base_path
    if train_df is None:
        base_path = _resolve_base_path_from_config()
        print(f"preprocess_data: não encontrei DataFrame nos 'dataframes' passados. Vou procurar ficheiros em: {base_path}")

        # procurar ficheiros com nomes comuns e extensões .csv/.xlsx/.xls
        cand_files = []
        patterns = ['boneage-training-dataset', 'boneage-training', 'training', 'train', 'boneage']
        for p in patterns:
            for ext in ('.csv', '.xlsx', '.xls'):
                cand = os.path.join(base_path, p + ext)
                if os.path.isfile(cand):
                    cand_files.append(cand)
        # se ainda nada, procura qualquer ficheiro com "training" no nome
        if not cand_files:
            for ext in ('csv','xlsx','xls'):
                cand_files.extend(glob.glob(os.path.join(base_path, f"*training*.{ext}")))
                cand_files.extend(glob.glob(os.path.join(base_path, f"*boneage*.{ext}")))

        if cand_files:
            # escolhe o primeiro plausível
            chosen = cand_files[0]
            print("preprocess_data: ficheiro encontrado:", chosen)
            try:
                if chosen.lower().endswith(('.xlsx', '.xls')):
                    train_df = _pd.read_excel(chosen)
                else:
                    train_df = _pd.read_csv(chosen)
                # tenta encontrar labels_df se houver outro ficheiro de labels
                # procura ficheiro test/train separadamente
                # labels_df permanecerá None se o ficheiro não tiver coluna label
                labels_df = None
            except Exception as e:
                raise RuntimeError(f"Falha a ler o ficheiro {chosen}: {e}")
        else:
            # não encontrou ficheiro no base_path: falha informativa
            raise RuntimeError(f"Não encontrei um DataFrame de treino nem ficheiros no base_path ({_resolve_base_path_from_config()}).")
    else:
        # se encontramos train_df a partir de dataframes, base_path pode ser resolvido também
        base_path = _resolve_base_path_from_config()

    # 3) agora normalizar nomes: encontrar coluna de filename e possivel coluna de label
    # procura coluna filename-like
    id_col = None
    for c in train_df.columns:
        if _is_filename_column(c):
            id_col = c
            break
    # se não existir coluna filename, tenta inferir por primeira coluna textual
    if id_col is None:
        for c in train_df.columns:
            if train_df[c].dtype == object:
                id_col = c
                print(f"preprocess_data: inferi coluna de filenames como '{id_col}'")
                break

    if id_col is None:
        raise KeyError("O DataFrame de treino não contém uma coluna de nomes de ficheiros identificável (id/filename).")

    # normalizar extensão nos ids
    train_df[id_col] = train_df[id_col].astype(str).apply(lambda x: x if str(x).lower().endswith((".png", ".jpg", ".jpeg")) else f"{x}.png")

    # procurar label col
    label_col = None
    for c in train_df.columns:
        if _is_label_column(c):
            label_col = c
            break
    # se labels n estiverem no mesmo DF, mas labels_df foi encontrado, tentar merge
    if label_col is None and labels_df is not None:
        id_col_labels = None
        for c in labels_df.columns:
            if _is_filename_column(c):
                id_col_labels = c
                break
        label_col_labels = None
        for c in labels_df.columns:
            if _is_label_column(c):
                label_col_labels = c
                break
        if id_col_labels and label_col_labels:
            labels_df[id_col_labels] = labels_df[id_col_labels].astype(str).apply(
                lambda x: x if str(x).lower().endswith((".png", ".jpg", ".jpeg")) else f"{x}.png"
            )
            train_df[id_col] = train_df[id_col].astype(str)
            merged = train_df.merge(labels_df[[id_col_labels, label_col_labels]],
                                    left_on=id_col, right_on=id_col_labels, how='left')
            train_df = merged
            label_col = label_col_labels

    # se label_col existe, normaliza para 'boneage' e cria bone_age_z
    mean_bone_age = None
    std_bone_age = None
    if label_col is not None and label_col in train_df.columns:
        if label_col != 'boneage':
            train_df = train_df.rename(columns={label_col: 'boneage'})
        mean_bone_age = float(train_df['boneage'].mean())
        std_bone_age = float(train_df['boneage'].std())
        train_df['bone_age_z'] = (train_df['boneage'] - mean_bone_age) / std_bone_age

    has_labels = ('bone_age_z' in train_df.columns)

    # 4) split treino/val (só se tivermos labels)
    if has_labels:
        df_train, df_valid = train_test_split(train_df, test_size=0.2, random_state=0)
    else:
        df_train = train_df.copy()
        df_valid = pd.DataFrame(columns=train_df.columns)

    # 5) localizar a pasta de imagens
    train_img_dir, test_img_dir = _guess_image_dirs(base_path)
    # diagnostic: report which image directories were guessed
    print(f"preprocess_data: train_img_dir={train_img_dir}, test_img_dir={test_img_dir}")

        # se não encontrou train_img_dir tenta supor que as imagens estão numa folder nomeada como no CSV
        # --- encontrar train_img_dir se ainda não foi encontrado ---
    if train_img_dir is None:
        for root, dirs, files in os.walk(base_path):
            imgs = {f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))}
            sample_ids = set(df_train['id'].astype(str).tolist()[:50])
            if len(sample_ids & imgs) >= 5:
                train_img_dir = root
                break

    if train_img_dir is None:
        raise FileNotFoundError("Não foi possível localizar a pasta de imagens de treino a partir do data_access config e heurísticas.")

    # Garantir que found_count existe sempre e calcular quantos ficheiros do CSV existem
    img_listed_ids = df_train['id'].astype(str).tolist()
    found_count = 0
    if train_img_dir and os.path.isdir(train_img_dir):
        try:
            found_count = sum(1 for fn in img_listed_ids if os.path.exists(os.path.join(train_img_dir, fn)))
        except Exception:
            found_count = 0
    else:
        found_count = 0

    print(f"preprocess_data: encontrei {found_count}/{len(img_listed_ids)} ficheiros listados na pasta {train_img_dir}")

    # --- construir generators ---
    img_size = 256
    batch_size = 32

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       rotation_range=10, zoom_range=0.1,
                                       width_shift_range=0.05, height_shift_range=0.05)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Se houver labels, class_mode='other' (regression). Caso contrário, class_mode=None
    class_mode_train = 'other' if has_labels else None
    class_mode_val = 'other' if has_labels else None

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=train_img_dir,
        x_col='id',
        y_col='bone_age_z' if has_labels else None,
        target_size=(img_size, img_size),
        color_mode='rgb',
        class_mode=class_mode_train,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    val_generator = None
    if not df_valid.empty:
        val_generator = val_datagen.flow_from_dataframe(
            dataframe=df_valid,
            directory=train_img_dir,
            x_col='id',
            y_col='bone_age_z' if has_labels else None,
            target_size=(img_size, img_size),
            color_mode='rgb',
            class_mode=class_mode_val,
            batch_size=batch_size,
            shuffle=False
        )

    # --- criar test_generator correctamente quando as imagens estão directamente na pasta ---
    test_generator = None
    if test_img_dir and os.path.isdir(test_img_dir):
        files_in_test = [f for f in os.listdir(test_img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if len(files_in_test) > 0:
            # cria DataFrame listing as imagens para usar flow_from_dataframe (class_mode=None)
            test_df = pd.DataFrame({"id": files_in_test})
            test_generator = test_datagen.flow_from_dataframe(
                dataframe=test_df,
                directory=test_img_dir,
                x_col='id',
                y_col=None,
                target_size=(img_size, img_size),
                color_mode='rgb',
                class_mode=None,
                batch_size=batch_size,
                shuffle=False
            )
        else:
            # fallback: usar flow_from_directory caso haja subpastas com imagens
            test_generator = test_datagen.flow_from_directory(
                directory=test_img_dir,
                target_size=(img_size, img_size),
                color_mode='rgb',
                class_mode=None,
                batch_size=batch_size,
                shuffle=False
            )

    # --- criar dict serializável a gravar (excluir objetos não pickláveis como generators) ---
    processed_data_to_pickle = {
        "df_train": df_train,
        "df_valid": df_valid,
        "mean_bone_age": mean_bone_age,
        "std_bone_age": std_bone_age,
        "train_img_dir": train_img_dir,
        "test_img_dir": test_img_dir,
        "img_size": img_size,
        "batch_size": batch_size
        # NÃO incluir train_generator/val_generator/test_generator
    }

    # grava só o que é serializável
    with open(_PROCESSED_PICKLE, "wb") as f:
        pickle.dump(processed_data_to_pickle, f)

    # --- manter em memória os generators e devolver processed_data completo ---
    processed_data = {
        "df_train": df_train,
        "df_valid": df_valid,
        "mean_bone_age": mean_bone_age,
        "std_bone_age": std_bone_age,
        "train_img_dir": train_img_dir,
        "test_img_dir": test_img_dir,
        "train_generator": train_generator,
        "val_generator": val_generator,
        "test_generator": test_generator,
        "img_size": img_size,
        "batch_size": batch_size
    }

    return processed_data
