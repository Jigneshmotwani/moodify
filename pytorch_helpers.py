import torch
import os
from python_helpers import get_project_root_dir

feature_dir = os.path.join(get_project_root_dir(), 'processed', 'features')
label_dir = os.path.join(get_project_root_dir(), 'processed', 'label')

def save_processed_data(features: dict[str, torch.Tensor], labels: torch.Tensor, dset_name: str):
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    fn = dset_name + '.pt'
    torch.save(features, os.path.join(feature_dir, fn))
    torch.save(labels, os.path.join(label_dir, fn))

    print(f'{dset_name} data saved successfully!')

def load_data(dset_name: str):
    fn = dset_name + '.pt'
    features = torch.load(os.path.join(feature_dir, fn))
    labels = torch.load(os.path.join(label_dir, fn))

    return features, labels

def load_all_data():
    feature_paths = sorted([os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if os.path.splitext(f)[1] == '.pt'])
    label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if os.path.splitext(f)[1] == '.pt'])

    feature_ls = [torch.load(p) for p in feature_paths]
    feature_names = list(feature_ls[0].keys())
    features = {k: torch.cat([d[k] for d in feature_ls], dim=0) for k in feature_names}
    labels = torch.cat([torch.load(p) for p in label_paths], dim=0)

    return features, labels
