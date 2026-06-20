import io
import json
import os
import re
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


@dataclass
class ATNSubjectSample:
    subject_id: str
    target: np.ndarray
    mask: np.ndarray
    age: float
    label: int


class ATNTorchDataset(Dataset):
    """Torch dataset for ATN evaluation."""

    def __init__(self, samples: List[ATNSubjectSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.from_numpy(sample.target).float(),
            torch.from_numpy(sample.mask).float(),
            torch.tensor(sample.age, dtype=torch.float32),
            torch.tensor(sample.label, dtype=torch.long),
        )


class ATNTrainDataset(Dataset):
    """Torch dataset for ATN training (no GluonTS transforms)."""

    def __init__(self, samples: List[ATNSubjectSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = torch.from_numpy(sample.target).float()
        mask = torch.from_numpy(sample.mask).float()
        age = torch.tensor(sample.age, dtype=torch.float32)
        label = torch.tensor(sample.label, dtype=torch.long)

        return {
            "target": target,
            "mask": mask,
            "age": age,
            "label": label,
            "subject_id": sample.subject_id,
        }


def _load_json(json_file: str):
    with open(json_file, "r") as f:
        return json.load(f)


def _process_amyloid(file_path: str):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r") as file:
            content = file.read()
            match = re.search(r"# ColHeaders\s+Index", content)
            if not match:
                return None
            table_start = match.end()
            table_data = content[table_start:].strip()

            col_separator = r"\s+"
            column_names = [
                "Index",
                "SegId",
                "NVoxels",
                "Volume_mm3",
                "StructName",
                "Mean",
                "StdDev",
                "Min",
                "Max",
                "Range",
            ]
            data = pd.read_csv(io.StringIO(table_data), sep=col_separator, names=column_names, engine="python")

            row_index = data[data["SegId"] == "11101"].index
            if len(row_index) > 0:
                data = data.iloc[row_index[0] :]
                data = data.drop("Index", axis=1)

            cols = ["SegId", "NVoxels", "Volume_mm3", "Mean", "StdDev", "Min", "Max", "Range"]
            data[cols] = data[cols].apply(pd.to_numeric, errors="coerce")

            if len(data) > 148:
                data = data.iloc[-148:]

            return data[["SegId", "Mean"]]
    except Exception:
        return None


def _process_ctx(file_path: str, json_dict):
    if not os.path.exists(file_path):
        return None
    try:
        data = loadmat(file_path)
        a = data["S"]["aparc_a2009s"][0][0][0][0][1][:].flatten()
        b = data["S"]["aparc_a2009s"][0][0][0][0][4][0][0][0].flatten()

        data = pd.DataFrame({"SegId": a, "value": b})
        data = data.iloc[2:]
        if 84 in data.index:
            data = data.drop(84)
        if 85 in data.index:
            data = data.drop(85).reset_index(drop=True)

        def map_to_aal_id(segid):
            if isinstance(segid, np.ndarray):
                segid = segid[0]
            segid = str(segid).strip("[]'")
            side = "lh" if segid.startswith("l") else "rh"
            core_name = segid[1:]
            for entry in json_dict:
                name = entry.get("name", "")
                if name.startswith(f"ctx_{side}_") and name[7:] == core_name:
                    return entry.get("AAL_ID", None)
            return None

        data["AAL_ID"] = data["SegId"].apply(lambda x: map_to_aal_id(x))
        data["AAL_ID"] = pd.to_numeric(data["AAL_ID"], errors="coerce")
        df_sorted = data.sort_values(by="AAL_ID").dropna(subset=["AAL_ID"]).reset_index(drop=True)

        return df_sorted[["SegId", "value"]]
    except Exception:
        return None


def _process_subject(subject_path: str, json_dict) -> List[dict]:
    subject_data = []
    time_folders = sorted(os.listdir(subject_path))

    for time_folder in time_folders:
        time_path = os.path.join(subject_path, time_folder)
        if not os.path.isdir(time_path):
            continue

        beta_file_path = os.path.join(time_path, "beta")
        tau_file_path = os.path.join(time_path, "tau")
        ctx_file_path = os.path.join(time_path, "catROIs_t1.mat")

        amyloid_df = _process_amyloid(beta_file_path)
        tau_df = _process_amyloid(tau_file_path)
        ctx_df = _process_ctx(ctx_file_path, json_dict)

        if amyloid_df is None or tau_df is None or ctx_df is None:
            continue

        amy_val = amyloid_df["Mean"].values
        tau_val = tau_df["Mean"].values
        ctx_val = ctx_df["value"].values

        label_path = os.path.join(time_path, "label.pt")
        age_path = os.path.join(time_path, "age.pt")

        if not os.path.exists(label_path) or not os.path.exists(age_path):
            continue

        label = torch.load(label_path)
        age = torch.load(age_path)

        combined_data = {
            "amyloid": amy_val,
            "tau": tau_val,
            "ctx": ctx_val,
            "age": age,
            "label": label,
            "subject_id": os.path.basename(subject_path),
        }
        subject_data.append(combined_data)

    return subject_data


def _load_atn_samples(data_root: str, json_file: str, prediction_length: int) -> List[ATNSubjectSample]:
    json_dict = _load_json(json_file)
    all_samples: List[ATNSubjectSample] = []

    if not os.path.exists(data_root):
        return all_samples

    for type_folder in sorted(os.listdir(data_root)):
        type_path = os.path.join(data_root, type_folder)
        if not os.path.isdir(type_path):
            continue

        for subject_folder in sorted(os.listdir(type_path)):
            subject_path = os.path.join(type_path, subject_folder)
            if not os.path.isdir(subject_path):
                continue

            subject_data = _process_subject(subject_path, json_dict)
            if not subject_data:
                continue

            raw_amyloid = np.array([item["amyloid"] for item in subject_data])
            raw_tau = np.array([item["tau"] for item in subject_data])
            raw_ctx = np.array([item["ctx"] for item in subject_data])

            biomarkers = np.concatenate([raw_amyloid, raw_tau, raw_ctx], axis=1)
            num_time_points = biomarkers.shape[0]

            raw_age = subject_data[0]["age"].item() if isinstance(subject_data[0]["age"], torch.Tensor) else subject_data[0]["age"]
            raw_label = subject_data[0]["label"].item() if isinstance(subject_data[0]["label"], torch.Tensor) else subject_data[0]["label"]

            if num_time_points < prediction_length:
                pad_len = prediction_length - num_time_points
                pad = np.full((pad_len, biomarkers.shape[1]), np.nan)
                biomarkers = np.concatenate([biomarkers, pad], axis=0)
            elif num_time_points > prediction_length:
                biomarkers = biomarkers[:prediction_length]

            mask = np.isfinite(biomarkers).all(axis=1).astype(np.float32)
            target = biomarkers.T

            all_samples.append(
                ATNSubjectSample(
                    subject_id=os.path.basename(subject_path),
                    target=target,
                    mask=mask,
                    age=float(raw_age),
                    label=int(raw_label),
                )
            )

    return all_samples


def build_atn_samples(
    train_path: str,
    test_path: str,
    json_file: str,
    prediction_length: int,
):
    train_samples = _load_atn_samples(train_path, json_file, prediction_length)
    test_samples = _load_atn_samples(test_path, json_file, prediction_length)
    return train_samples, test_samples
