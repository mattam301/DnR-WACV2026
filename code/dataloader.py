import torch
import math
import numpy as np
import random
import pickle
import os
from tqdm import tqdm


DATASET_ALIASES = {
    "iemocap_coid": "iemocap",
    "meld_coid": "meld",
    "mosi_coid": "mosi",
    "mosei_coid": "mosei",
    "humor_coid": "humor",
    "sarcasm_coid": "sarcasm",
}


AFFECT_DATASETS = {
    "mosi": {
        "path": "data/mosi/mosi_data.pkl",
        "file_id": "1_XdzdW8UNG1TTS6QcX10uhoS6N11OBit",
    },
    "mosei": {
        "path": "data/mosei/mosei_senti_data.pkl",
        "file_id": "180l4pN6XAv8-OAYQ6OrMheFUMwtqUWbz",
    },
    "humor": {
        "path": "data/humor/humor.pkl",
        "file_id": "1L5slPmYyhEVtwGyM1kgcFMjeBpXLZGT0",
    },
    "sarcasm": {
        "path": "data/sarcasm/sarcasm.pkl",
        "file_id": "1EMBUmUL5B0PTncGx3L-sBElGOmjFBR_h",
    },
}


def base_dataset_name(dataset):
    return DATASET_ALIASES.get(dataset, dataset)

def load_iemocap():
    path = "data/iemocap/iemocap.pkl"
    with open(path, "rb") as f:
        unsplit = pickle.load(f)
    
    speaker_to_idx = {"M": 0, "F": 1}

    data = {
        "train": [], "dev": [], "test": [],
    }
    trainVid = list(unsplit["trainVid"])
    random.shuffle(trainVid)
    testVid = list(unsplit["testVid"])

    dev_size = int(len(trainVid) * 0.1)
    
    spliter = {
        "train": trainVid[dev_size:],
        "dev": trainVid[:dev_size],
        "test": testVid
    }
    
    for split in data:
        cur_len = 0
        for j, uid in tqdm(enumerate(spliter[split]), desc=split):
            data[split].append(
                {
                    "uid" : cur_len,
                    "speakers" : [speaker_to_idx[speaker] for speaker in unsplit["speaker"][uid]],
                    "labels" : unsplit["label"][uid],
                    "text": unsplit["text"][uid],
                    "audio": unsplit["audio"][uid],
                    "visual": unsplit["visual"][uid],
                    "sentence" : unsplit["sentence"][uid],
                }
            )
            cur_len += len(unsplit["speaker"][uid])
    return data

def load_meld():
    path = "data/meld/meld.pkl"
    with open(path, "rb") as f:
        unsplit = pickle.load(f)

    data = {
        "train": [], "dev": [], "test": [],
    }
    trainVid = list(unsplit["trainVid"])
    testVid = list(unsplit["testVid"])

    dev_size = int(len(trainVid) * 0.1)
    
    spliter = {
        "train": trainVid[dev_size:],
        "dev": trainVid[:dev_size],
        "test": testVid
    }

    spker = set()
    all_sp = []
    idx = 0
    for split in data:
        for j, uid in tqdm(enumerate(spliter[split]), desc=split):
            unsplit["speakers"][uid] = np.array([x if x != 8 else 7 for x in unsplit["speakers"][uid]])
            data[split].append(
                {
                    "uid" : j,
                    "speakers" : unsplit["speakers"][uid],
                    "labels" : unsplit["label"][uid],
                    "text": unsplit["text"][uid],
                    "audio": unsplit["audio"][uid],
                    "visual": unsplit["visual"][uid],
                    "sentence" : unsplit["sentence"][uid],
                }
            )
    
    return data

def load_mosei(emo="7class"):
    path = "data/mosei/mosei_data.pkl"
    with open(path, "rb") as f:
        unsplit = pickle.load(f)

    data = {
        "train": [], "dev": [], "test": [],
    }
    trainVid = list(unsplit["trainVid"])
    valVid = list(unsplit["valVid"])
    testVid = list(unsplit["testVid"])
    
    spliter = {
        "train": trainVid,
        "dev": valVid,
        "test": testVid
    }

    for split in data:
        for j, uid in tqdm(enumerate(spliter[split]), desc=split):
            data[split].append(
                {
                    "uid" : j,
                    "speakers" : [0] * len(unsplit["speaker"][uid]),
                    "labels" : unsplit['label'][emo][uid],
                    "text": unsplit["text"][uid],
                    "audio": unsplit["audio"][uid],
                    "visual": unsplit["visual"][uid],
                    "sentence" : unsplit["sentence"][uid],
                }
            )
    
    return data


def _download_affect_dataset(name, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            f"{path} does not exist. Install gdown or download the {name} pickle "
            "with scripts/download_affect.sh."
        ) from exc
    url = f"https://drive.google.com/uc?id={AFFECT_DATASETS[name]['file_id']}"
    print(f"Downloading {name} to {path}")
    gdown.download(url, path, quiet=False)


def _first_scalar(value):
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        raise ValueError("Empty label encountered")
    return float(arr[0])


def _binary_label(value):
    return int(_first_scalar(value) > 0)


def _valid_text_mask(text):
    text = np.asarray(text)
    if text.ndim == 1:
        return np.abs(text) > 0
    return np.any(np.abs(text) > 0, axis=-1)


def _pool_by_mask(features, mask):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        return features
    if len(mask) == features.shape[0] and np.any(mask):
        features = features[mask]
    return features.mean(axis=0).astype(np.float32)


def load_affect_dataset(name):
    """Load MultiBench affect datasets into this repo's dialogue-shaped format.

    MOSI, MOSEI, UR-FUNNY, and MUSTARD are clip-level tasks. To keep the
    original MERC training code unchanged, each clip is represented as a
    one-utterance dialogue whose utterance features are mean-pooled over the
    valid text-aligned timesteps.
    """
    name = base_dataset_name(name)
    if name not in AFFECT_DATASETS:
        raise ValueError(f"Unsupported affect dataset: {name}")

    path = AFFECT_DATASETS[name]["path"]
    if not os.path.exists(path):
        _download_affect_dataset(name, path)

    with open(path, "rb") as f:
        unsplit = pickle.load(f)

    split_map = {"train": "train", "dev": "valid", "test": "test"}
    data = {"train": [], "dev": [], "test": []}
    for out_split, in_split in split_map.items():
        split_data = unsplit[in_split]
        n_samples = len(split_data["text"])
        for j in tqdm(range(n_samples), desc=out_split):
            text = np.asarray(split_data["text"][j], dtype=np.float32)
            mask = _valid_text_mask(text)
            sample_id = split_data.get("id", [j] * n_samples)[j]
            data[out_split].append(
                {
                    "uid": j,
                    "speakers": [0],
                    "labels": [_binary_label(split_data["labels"][j])],
                    "text": [_pool_by_mask(text, mask)],
                    "audio": [_pool_by_mask(split_data["audio"][j], mask)],
                    "visual": [_pool_by_mask(split_data["vision"][j], mask)],
                    "sentence": [str(sample_id)],
                }
            )
    return data


def infer_feature_dims(data):
    sample = data["train"][0]
    return {
        "t": len(np.asarray(sample["text"][0]).reshape(-1)),
        "a": len(np.asarray(sample["audio"][0]).reshape(-1)),
        "v": len(np.asarray(sample["visual"][0]).reshape(-1)),
    }

class Dataloader:
    def __init__(self, data, args):
        self.data = data
        self.batch_size = args.batch_size
        self.num_batches = math.ceil(len(data)/ self.batch_size)
        self.dataset = args.dataset
        self.embedding_dim = args.input_embedding_dim[self.dataset]
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.data[index * self.batch_size : (index + 1) * self.batch_size]
        return batch

    def padding(self, samples):
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s["text"]) for s in samples]).long()
        uid = torch.tensor([s["uid"] for s in samples]).long()
        mx = torch.max(text_len_tensor).item()
        
        audio_tensor = torch.zeros((batch_size, mx, self.embedding_dim['a']))
        text_tensor = torch.zeros((batch_size, mx, self.embedding_dim['t']))
        visual_tensor = torch.zeros((batch_size, mx, self.embedding_dim['v']))
        speaker_tensor = torch.zeros((batch_size, mx)).long()

        labels = []
        utterances = []
        for i, s in enumerate(samples):
            cur_len = len(s["text"])
            utterances.append(s["sentence"])

            tmp_t = []
            tmp_a = []
            tmp_v = []
            for t, a, v in zip(s["text"], s["audio"], s["visual"]):
                tmp_t.append(torch.tensor(t))
                tmp_a.append(torch.tensor(a))
                tmp_v.append(torch.tensor(v))
                
            tmp_a = torch.stack(tmp_a)
            tmp_t = torch.stack(tmp_t)
            tmp_v = torch.stack(tmp_v)

            text_tensor[i, :cur_len, :] = tmp_t
            audio_tensor[i, :cur_len, :] = tmp_a
            visual_tensor[i, :cur_len, :] = tmp_v
            
            speaker_tensor[i, :cur_len] = torch.tensor(s["speakers"])

            labels.extend(s["labels"])

        label_tensor = torch.tensor(labels).long()
        

        data = {
            "uid": uid,
            "length": text_len_tensor,
            "tensor": {
                "t": text_tensor,
                "a": audio_tensor,
                "v": visual_tensor,
            },
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "utterance_texts": utterances,
        }

        return data

    def shuffle(self):
        random.shuffle(self.data)

