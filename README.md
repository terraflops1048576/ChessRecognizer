### Data
Download the data [here](https://drive.google.com/file/d/14D9diuX9xrP3HM8XmcKRU7-XcLI4vsE8/view?usp=sharing)

### Model architecture
The model uses a pretrained EfficientNet as a feature extractor from the chessboard images and then feeds this into the memory of several stacked Transformer decoder layers, which take in the [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) representation of the chess position as the text target.

### Project structure
The dataset implementation is in data.py, the model implementation is in model.py, and all of the training code is in train.py.
Note that running train.py will do nothing but load the data. To actually train the model, import trainer and train_dataloader from train.py and run
```
from train import trainer, evaluator, recognizer_full, train_dataloader, val_dataloader
trainer.run(train_dataloader, max_epochs=2) # trains the model
```

### Current debugging efforts
After training for a few epochs, the model becomes completely indifferent to the text input used to generate a sequence, in that its output probabilities do not change at all, no matter the currently generated sequence.
It is incapable of even overfitting on batches of copies of the first sample in the dataset.