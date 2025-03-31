import torch
import pytorch_helpers as pth
import sklearn.model_selection as skl

# class MelSpecDataset(torch.utils.data.Dataset):
class CustomDataset(torch.utils.data.Dataset):
    # def __init__(self, melspecs: torch.Tensor, labels: torch.Tensor):
    def __init__(self, features: dict[str, torch.Tensor], labels: torch.Tensor):
        # super(MelSpecDataset, self).__init__()
        super(CustomDataset, self).__init__()

        for k,v in features.items():
            if v.shape[0] != labels.shape[0]:
                raise ValueError(f'Feature {k} tensor length does not match label tensor length ({v.shape[0]} vs {labels.shape[0]})')
            
        self.features = features
        self.labels = labels

        # self.waveforms = features['waveforms']
        # self.spectrograms = features['spectrograms']
        # self.melspecs = features['melspecs']
        # self.mfcc = features['mfcc']
        # Assumption: melspecs has shape (N,H,W) or (N,C,H,W)
        # assert melspecs.shape[0] == labels.shape[0] and 3 <= len(melspecs.shape) <= 4
        # self.melspecs = melspecs
        # self.labels = labels

    @property
    def waveforms(self):
        return self.features['waveforms']

    @property
    def spectrograms(self):
        return self.features['spectrograms']

    @property
    def melspecs(self):
        return self.features['melspecs']

    @property
    def mfcc(self):
        return self.features['mfcc']

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx: int):
        # return self.melspecs[idx, ...], self.labels[idx]
        return {k: v[idx] for k,v in self.features.items()}, self.labels[idx]

    # def NCHW_to_NHW(self):
    #     if len(self.melspecs.shape) == 4 and self.melspecs.shape[1] == 1:
    #         self.melspecs = self.melspecs.squeeze(dim=1)
    #     else:
    #         print('Conversion to NHW failed: melspecs is not a 4D tensor OR size of dim 1 is not 1')
        
    #     return self

    # def NHW_to_NCHW(self):
    #     if len(self.melspecs.shape) == 3:
    #         self.melspecs = self.melspecs.unsqueeze(dim=1)
    #     else:
    #         print('Conversion to NCHW failed: melspecs is not a 3D tensor')
        
    #     return self
    
    def train_test_split(self, split_size: int|float=0.2):
        # train_melspecs, test_melspecs, train_labels, test_labels = skl.train_test_split(self.melspecs, self.labels, test_size=split_size, shuffle=True)

        # return MelSpecDataset(train_melspecs, train_labels), MelSpecDataset(test_melspecs, test_labels)

        indices = list(range(self.labels.shape[0]))
        train_indices, test_indices = skl.train_test_split(indices, test_size=split_size, shuffle=True)

        return (
            CustomDataset({k: v[train_indices] for k,v in self.features.items()}, self.labels[train_indices]),
            CustomDataset({k: v[test_indices] for k,v in self.features.items()}, self.labels[test_indices])
        )

    def to(self, device: torch.device):
        for k,v in self.features.items():
            self.features[k] = v.to(device)
        self.labels = self.labels.to(device)

        return self
    
    @property
    def device(self):
        return torch.device('cuda') if self.labels.is_cuda else torch.device('cpu')


# class AllDataset(MelSpecDataset):
#     def __init__(self):
#         melspecs, labels = pth.load_all_data()
#         super(AllDataset, self).__init__(melspecs, labels)
    
# class SoundTracksDataset(MelSpecDataset):
#     def __init__(self):
#         melspecs, labels = pth.load_data("SoundTracks")
#         super(SoundTracksDataset, self).__init__(melspecs, labels)

class AllDataset(CustomDataset):
    def __init__(self):
        features, labels = pth.load_all_data()
        super(AllDataset, self).__init__(features, labels)

class SoundTracksDataset(CustomDataset):
    def __init__(self):
        features, labels = pth.load_data("SoundTracks")
        super(SoundTracksDataset, self).__init__(features, labels)