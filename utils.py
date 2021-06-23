"""
TODO: add file description
"""
import torch
import os


class ModelManager:
    def __init__(self, train_len, val_len, output_dir):
        self.train_losses = []
        self.val_losses = []
        self.steps = []
        self.best_val_loss = float('Inf')
        self.train_len = train_len
        self.val_len = val_len
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
        print('Epoch [{}/{}], step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
              .format(epoch + 1, num_epochs, step, num_epochs * self.train_len, train_loss, val_loss))

        # checkpoint
        if self.best_val_loss > val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(save_as, model, self.best_val_loss)
            self.save_metrics(metric_file)
