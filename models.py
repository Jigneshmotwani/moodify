import torch
import torchaudio
from abc import ABC, abstractmethod
from transformers import WhisperForAudioClassification, WhisperProcessor


class MyModel(torch.nn.Module, ABC):
    @abstractmethod
    def __init__(self, feature: str):
        super(MyModel, self).__init__()
        valid_features = ['waveforms', 'spectrograms', 'melspecs', 'mfcc']
        if feature not in valid_features:
            raise ValueError(
                f'Feature name {feature} is not one of {valid_features}')
        self.feature = feature

    @abstractmethod
    def forward(self, features: dict[str, torch.Tensor]):
        return features

    @property
    def device(self):
        return torch.device('cuda') if next(self.parameters()).is_cuda else torch.device('cpu')


class NilsHMeierCNN(MyModel):
    """
    Implementation adapted from [Melspectrogram based CNN Classification by NilsHMeier on Kaggle](https://www.kaggle.com/code/nilshmeier/melspectrogram-based-cnn-classification)
    """

    def __init__(self, feature: str):
        super(NilsHMeierCNN, self).__init__(feature)

        self.pipeline = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16,
                            kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # torch.nn.Dropout2d(p=0.3),
            torch.nn.Conv2d(in_channels=16, out_channels=32,
                            kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Flatten(),
            torch.nn.Dropout1d(p=0.3),

            torch.nn.LazyLinear(out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=4),
        )

    def forward(self, features: dict[str, torch.Tensor]):
        x = features[self.feature].unsqueeze(dim=1)
        x = self.pipeline(x)

        return x    # softmax will be applied in loss function


class VGGStyleCNN(MyModel):
    """VGG-inspired CNN with batch normalization for 128x323 melspectrograms"""

    def __init__(self, feature: str):
        super().__init__(feature)

        self.cnn = torch.nn.Sequential(
            # Block 1
            torch.nn.Conv2d(1, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            # Block 2
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            # Block 3
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 16 * 40, 512),  # Adjusted for 128x323 input
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 4)
        )

    def forward(self, features: dict[str, torch.Tensor]):
        x = features[self.feature].unsqueeze(1)  # Add channel dimension
        x = self.cnn(x)
        return self.classifier(x)


class ResNetBlock(torch.nn.Module):
    """Basic ResNet block with skip connection"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)
    
    


class ResNetStyleCNN(MyModel):
    """ResNet-inspired architecture for audio classification"""

    def __init__(self, feature: str):
        super().__init__(feature)

        self.initial = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 4)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = [ResNetBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def forward(self, features: dict[str, torch.Tensor]):
        x = features[self.feature].unsqueeze(1)  # Add channel dimension
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(x)


class YAMNetMER(MyModel):
    """YAMNet-based Music Emotion Recognition"""
    def __init__(self, feature: str = 'waveforms'):
        super().__init__(feature)
        self.yamnet = torchaudio.pipelines.YAMNET.get_model()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 4)
        )
        
        for param in self.yamnet.parameters():
            param.requires_grad = False

    def forward(self, features: dict[str, torch.Tensor]):
        x = features[self.feature]
        with torch.no_grad():
            embeddings, _ = self.yamnet(x)
        return self.classifier(embeddings.mean(dim=1))

class EvoCNN(MyModel):
    """Evolutionary Optimized CNN Architecture"""
    def __init__(self, feature: str = 'melspecs'):
        super().__init__(feature)
        
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(2),
            
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(2),
            
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ELU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            
            torch.nn.Flatten(),
            torch.nn.Linear(128*4*4, 256),
            torch.nn.ELU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 4)
        )

    def forward(self, features: dict[str, torch.Tensor]):
        x = features[self.feature].unsqueeze(1)
        return self.model(x)

class WhisperMER(MyModel):
    """Fine-tuned Whisper-Large-v3 for Emotion Recognition"""
    def __init__(self, feature: str = 'whisper_inputs'):
        super().__init__(feature)
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.model = WhisperForAudioClassification.from_pretrained(
            "openai/whisper-large-v3",
            num_labels=4,
            problem_type="multi_class_classification"
        )
        
        # Freeze encoder except last 4 layers
        for param in self.model.encoder.parameters()[:-4]:
            param.requires_grad = False

    def forward(self, features: dict[str, torch.Tensor]):
        return self.model(**features).logits