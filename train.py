import os
import datetime

import matplotlib.pyplot as plt

import torch.utils.data.dataloader
import torch
import torchaudio
import torchvision
import torchmetrics
import shutil

import python_helpers as pyh
import pytorch_helpers as pth
import models
import datasets


class ModelTrainer:
    def __init__(self, task: str, num_classes: int, device: torch.device):
        self.task = task
        self.num_classes = num_classes
        self.device = device

        self.loss = torch.nn.CrossEntropyLoss().to(device)
        self.accuracy = torchmetrics.Accuracy(
            task=task, num_classes=num_classes).to(device)

    def train(self, model: torch.nn.Module, train_dset: datasets.CustomDataset, val_dset: datasets.CustomDataset, batch_size=8, max_epochs=5, lr=1e-3, lambda_val=0.0, l1_ratio=0.0, take_best=False):
        orig_device = model.device

        train_dset = train_dset.to(self.device)
        val_dset = val_dset.to(self.device)

        train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model = model.to(self.device)
        model.train()

        self.loss_history = {'train': [], 'val': []}
        self.accuracy_history = {'train': [], 'val': []}

        print("Training start")

        this_run_dir = os.path.join(pyh.get_project_root_dir(
        ), 'runs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        os.makedirs(this_run_dir, exist_ok=True)

        if take_best:
            best_epoch = 0
            best_model_path = os.path.join(this_run_dir, 'best.pt')
            self.save_model(model, best_model_path)
        final_model_path = os.path.join(this_run_dir, 'final.pt')

        init_train_loss, init_train_acc, _ = self.evaluate_performance(
            model, train_dset, lambda_val, l1_ratio)
        init_val_loss, init_val_acc, _ = self.evaluate_performance(
            model, val_dset, lambda_val, l1_ratio)

        print(
            f'Epoch 0    Train Loss={init_train_loss:.4f}    Train Acc={init_train_acc:.4f}    Val Loss={init_val_loss:.4f}    Val Acc={init_val_acc:.4f}')

        self.loss_history['train'].append(init_train_loss)
        self.accuracy_history['train'].append(init_train_acc)
        self.loss_history['val'].append(init_val_loss)
        self.accuracy_history['val'].append(init_val_acc)

        for i in range(max_epochs):
            model.train()

            num_batches = len(train_loader)
            for features, labels in train_loader:

                optimizer.zero_grad()
                preds = model(features)
                loss = self.loss(preds, labels) + lambda_val * \
                    self.regularisation(model, l1_ratio) / num_batches

                loss.backward()
                optimizer.step()

            epoch_train_loss, epoch_train_acc, _ = self.evaluate_performance(
                model, train_dset, lambda_val, l1_ratio)
            epoch_val_loss, epoch_val_acc, _ = self.evaluate_performance(
                model, val_dset, lambda_val, l1_ratio)

            if take_best:
                if (
                    epoch_val_acc > self.accuracy_history['val'][best_epoch]
                    # and epoch_val_acc > (epoch_train_acc - 0.01)
                ) or (
                    epoch_val_acc == self.accuracy_history['val'][best_epoch]
                    and epoch_train_acc >= self.accuracy_history['train'][best_epoch]
                    # and epoch_val_acc > (epoch_train_acc - 0.01)
                ):
                    self.save_model(model, best_model_path)
                    best_epoch = i+1

            self.loss_history['train'].append(epoch_train_loss)
            self.accuracy_history['train'].append(epoch_train_acc)
            self.loss_history['val'].append(epoch_val_loss)
            self.accuracy_history['val'].append(epoch_val_acc)

            print(f'Epoch {i+1}    Train Loss={epoch_train_loss:.4f}    Train Acc={epoch_train_acc:.4f}    Val Loss={epoch_val_loss:.4f}    Val Acc={epoch_val_acc:.4f}')

        print("Saving final model...")
        self.save_model(model, final_model_path)

        if take_best:
            print(f'Loading best model from Epoch {best_epoch}...')
            self.load_model(model, best_model_path)
            self.best_epoch = best_epoch
        else:
            print("Loading final model...")
            if getattr(self, 'best_epoch', None) is not None:
                del self.best_epoch

        model = model.to(orig_device)
        model.eval()
        print('Training finished')

        self.save_histories_plot(os.path.join(this_run_dir, 'histories.png'))

    def predict(self, model: torch.nn.Module, dset: datasets.CustomDataset):
        model.eval()

        with torch.no_grad():
            preds = model(dset.features)
            labels = dset.labels

        return preds, labels

    def evaluate_accuracy(self, preds: torch.Tensor, labels: torch.Tensor):
        return self.accuracy(preds, labels).item()

    def evaluate_loss(self, model: torch.nn.Module, preds: torch.Tensor, labels: torch.Tensor, lambda_val=0.0, l1_ratio=0.0):
        return (self.loss(preds, labels).item() + lambda_val * self.regularisation(model, l1_ratio)).item()

    def confusion_matrix(self, preds: torch.Tensor, labels: torch.Tensor):
        return torchmetrics.ConfusionMatrix(task=self.task, num_classes=self.num_classes).to(torch.device('cuda' if preds.is_cuda else ('cpu')))(preds, labels)

    def evaluate_performance(self, model: torch.nn.Module, dset: datasets.CustomDataset, lambda_val=0.0, l1_ratio=0.0):
        preds, labels = self.predict(model, dset)
        loss = self.evaluate_loss(model, preds, labels, lambda_val, l1_ratio)
        accuracy = self.evaluate_accuracy(preds, labels)
        cm = self.confusion_matrix(preds, labels)

        return loss, accuracy, cm

    def save_histories_plot(self, dst_path):
        train_acc, val_acc = self._extract_history(hist_type='acc')
        train_loss, val_loss = self._extract_history(hist_type='loss')

        fig, ax = plt.subplots(2, 1)
        fig.set_figheight(12)
        fig.set_figwidth(20)

        idx = [i+1 for i in range(len(train_acc))]
        idx = range(len(train_acc))

        ax[0].plot(idx, train_loss, label='Train')
        ax[0].plot(idx, val_loss, label='Validation')
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epoch #')
        ax[0].set_ylabel('Loss')
        if getattr(self, 'best_epoch', None) is not None:
            ax[0].axvline(self.best_epoch, color='g',
                          linestyle=':', label='Best Epoch')
        ax[0].legend()

        ax[1].plot(idx, train_acc, '.-', label='Train')
        ax[1].plot(idx, val_acc, '.-', label='Validation')
        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('Epoch #')
        ax[1].set_ylabel('Accuracy')
        if getattr(self, 'best_epoch', None) is not None:
            ax[1].axvline(self.best_epoch, color='green',
                          linestyle=':', label='Best Epoch')
            ax[1].axhline(train_acc[self.best_epoch], color='blue',
                          linestyle=':', label='Best Epoch Train Acc.')
            ax[1].axhline(val_acc[self.best_epoch], color='orange',
                          linestyle=':', label='Best Epoch Val. Acc.')
        ax[1].legend()

        fig.savefig(dst_path, bbox_inches='tight', pad_inches=2)

    def _extract_history(self, hist_type: str = 'acc'):
        if hist_type == 'acc':
            history = self.accuracy_history
        elif hist_type == 'loss':
            history = self.loss_history

        train_hist = history['train']
        val_hist = history['val']

        return train_hist, val_hist

    def L1_reg(self, model: torch.nn.Module):
        return sum(torch.abs(param).sum() for param in model.parameters())

    def L2_reg(self, model: torch.nn.Module):
        return sum(torch.norm(param, p=2).sum() for param in model.parameters()) / 2

    def elasticnet_reg(self, model: torch.nn.Module, l1_ratio=0.0):
        return l1_ratio * self.L1_reg(model) + (1 - l1_ratio) * self.L2_reg(model)

    def regularisation(self, model: torch.nn.Module, l1_ratio=0.0):
        return self.elasticnet_reg(model, l1_ratio)

    def save_model(self, model: torch.nn.Module, dst_path: str):
        try:
            # Ensure model is on CPU before saving
            model = model.cpu()

            # Create parent directories if needed
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            # Use alternative serialization method
            torch.save(
                model.state_dict(),
                dst_path,
                _use_new_zipfile_serialization=False  # Disable zip format
            )
            print(f"Successfully saved to {os.path.abspath(dst_path)}")
        except Exception as e:
            print(f"Failed to save model: {str(e)}")
            raise

    def load_model(self, model: torch.nn.Module, src_path: str):
        model.load_state_dict(torch.load(src_path))


if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = datasets.SoundTracksDataset()

    num_split = len(dataset) // 10
    train_dset, test_dset = dataset.train_test_split(num_split)
    train_dset, val_dset = train_dset.train_test_split(num_split)
    print(len(train_dset), len(val_dset), len(test_dset))
    # exit()

    model = models.NilsHMeierCNN('melspecs')
    # print(model)

    trainer = ModelTrainer(task='multiclass', num_classes=4, device=device)

    # print(model.device, trainer.device, train_dset.device, val_dset.device, test_dset.device)

    bs = 16
    epochs = 100
    lam = 2.0
    l1_ratio = 0.02
    lr = 0.00005

    test_loss, test_acc, test_cm = trainer.evaluate_performance(
        model, test_dset, lambda_val=lam, l1_ratio=l1_ratio)
    print(f'Test Loss={test_loss:.4f}    Test Acc={test_acc:.4f}')
    print(test_cm)

    trainer.train(
        model, train_dset, val_dset, batch_size=bs,
        lr=lr, max_epochs=epochs, lambda_val=lam, l1_ratio=l1_ratio, take_best=True,
    )

    test_loss, test_acc, test_cm = trainer.evaluate_performance(
        model, test_dset, lambda_val=lam, l1_ratio=l1_ratio)
    print(f'Test Loss={test_loss:.4f}    Test Acc={test_acc:.4f}')
    print(test_cm)
