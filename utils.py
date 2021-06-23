"""
File name: utils.py
This file provides 2 main utilities:
1)  ModelManager class
    Model Manager is used to load and save model weights.
    It is also responsible for showing training progress and saving train and validation losses of the entire process.

2)  set_device function
    This function is used to safely set hardware accelerator to 'cuda' or 'cpu'
"""
import torch
import os


class ModelManager:
    def __init__(self, output_dir):
        self.train_losses = []
        self.val_losses = []
        self.steps = []
        self.best_val_loss = float('Inf')
        self.output_dir = output_dir

    def save_checkpoint(self, filename, model, valid_loss):
        torch.save({'model_state_dict': model.state_dict(), 'valid_loss': valid_loss},
                   os.path.join(self.output_dir, filename))

    def load_checkpoint(self, filename, model):
        state_dict = torch.load(os.path.join(self.output_dir, filename))
        model.load_state_dict(state_dict['model_state_dict'])
        return state_dict['valid_loss']

    def save_metrics(self, filename):
        state_dict = {'train_losses': self.train_losses,
                      'val_losses': self.val_losses,
                      'steps': self.steps}

        torch.save(state_dict, os.path.join(self.output_dir, filename))

    def load_metrics(self, filename, device):
        state_dict = torch.load(os.path.join(self.output_dir, filename), map_location=device)
        return state_dict['train_losses'], state_dict['val_losses'], state_dict['steps']

    def update_train_val_loss(self, model, train_loss, val_loss, step, epoch, num_epochs, save_as, metric_file):
        train_loss = train_loss
        val_loss = val_loss
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.steps.append(step)
        print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
              .format(epoch + 1, num_epochs, train_loss, val_loss))

        # checkpoint
        if self.best_val_loss > val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(save_as, model, self.best_val_loss)
            self.save_metrics(metric_file)


def set_device(dev):
    if dev == "cuda":
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            device = torch.device('cpu')
            print("CUDA is not available, using CPU...")
    else:
        device = torch.device('cpu')
    return device
