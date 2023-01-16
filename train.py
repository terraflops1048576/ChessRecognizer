import torch
import torchvision
import json
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from torch.utils.data import DataLoader
from data import ChessImageDataset

from model import ChessRecognizer, ChessRecognizerFull
from typing import Dict, List, Tuple, Any


class RowWiseAccuracy(Metric):
    def __init__(self):
        self.num_correct = 0
        self.total_images = 0
        super().__init__()
    
    @reinit__is_reduced
    def reset(self):
        self.num_correct = 0
        self.total_images = 0
        super().reset()

    @reinit__is_reduced
    def update(self, output):
        rowwise_accuracy, all_accuracy, n_tokens, n_images = output
        self.num_correct += round(rowwise_accuracy * n_images)
        self.total_images += n_images

    @sync_all_reduce("total_images", "num_correct:SUM")
    def compute(self):
        if self.total_images == 0:
            raise NotComputableError('Must have at least one image to be computed')
        return self.num_correct / self.total_images

class TokenAccuracy(Metric):
    def __init__(self):
        self.num_correct = 0
        self.total_tokens = 0
        super().__init__()
    
    @reinit__is_reduced
    def reset(self):
        self.num_correct = 0
        self.total_tokens = 0
        super().reset()

    @reinit__is_reduced
    def update(self, output):
        rowwise_accuracy, all_accuracy, n_tokens, n_images = output
        self.num_correct += round(all_accuracy * n_tokens)
        self.total_tokens += n_tokens

    @sync_all_reduce("total_tokens", "num_correct:SUM")
    def compute(self):
        if self.total_tokens == 0:
            raise NotComputableError('Must have at least one token to be computed')
        return self.num_correct / self.total_tokens


device = torch.device('cuda')
recognizer = ChessRecognizer(256, 4, 6).to(device)
CHARS = '%><pkqrbnPKQRBN12345678/acdefgh-9w0 ' # first 3 are pad, end, and start, respectively
chr2idx = {ch: i for i, ch in enumerate(CHARS)}

print("Making model")
recognizer_full = ChessRecognizerFull(chr2idx, recognizer, device=device).to(device)
#print(recognizer_full.embedding_size)
recognizer_full(torch.zeros(2, 3, 224, 224).to(device), ['pk'] * 2, max_len=90)
#recognizer_full.convert_text_to_tensor(['AB>'] * 4, 90)
recognizer_full.compute_loss(torch.zeros(2, 3, 224, 224).to(device), ['pk'] * 2, max_len=90)

print("Loading data")
with open("data/images.json") as f:
    data_dict = json.load(f)
dataset = ChessImageDataset(data_dict, 'data', max_images=50)
print(f"Loaded {len(dataset)} images")
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.05, 0.15], generator=torch.Generator().manual_seed(42))
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=128, shuffle=True)
small_data = [dataset[0]] * 64
small_dataloader = DataLoader(small_data, batch_size=64, shuffle=True)

optimizer = torch.optim.Adam(recognizer_full.parameters(), lr=0.001) # default is 0.001
def train_step(engine, batch):
    recognizer_full.train()
    optimizer.zero_grad()
    input_img, output_texts = batch[0].to(device), batch[1]
    input_img = input_img.float() / 255.0
    loss = recognizer_full.compute_loss(input_img, output_texts)
    loss.backward()
    optimizer.step()
    return loss.item()

def validation_step(engine, batch):
    recognizer_full.eval()
    with torch.no_grad():
        x, y = batch[0].to(device), batch[1]
        x = x.float() / 255.0
        return recognizer_full.compute_accuracy(x, y)

trainer = Engine(train_step)
evaluator = Engine(validation_step)
RowWiseAccuracy().attach(evaluator, "row_accuracy")
TokenAccuracy().attach(evaluator, "token_accuracy")

@trainer.on(Events.ITERATION_COMPLETED(every=100))
def log_training(engine):
    batch_loss = engine.state.output
    lr = optimizer.param_groups[0]['lr']
    e = engine.state.epoch
    n = engine.state.max_epochs
    i = engine.state.iteration
    print(f"Epoch {e}/{n} : {i} - batch loss: {batch_loss}, lr: {lr}")

@trainer.on(Events.EPOCH_COMPLETED(every=2))
def run_validation(engine):
    print("Running validation")
    state = evaluator.run(val_dataloader)
    print(f"Row accuracy: {state.metrics['row_accuracy']}, Token accuracy: {state.metrics['token_accuracy']}")
