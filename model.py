import torch
import torchvision
from typing import Dict, List, Tuple, Any
import math

class ChessRecognizer(torch.nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, ff_dim: int=1024, train_efficientnet: bool=False):
        super().__init__()
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.efficientnet = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier[1] = torch.nn.Identity() # Shaves off the classifier layer, which we don't want
        self.efficientnet.requires_grad_(train_efficientnet)
        self.linear = torch.nn.Linear(1280, d_model)
        self.activation = torch.nn.SiLU()
        self.linear2 = torch.nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.nhead = nhead
    
    def forward(self, input_img, generated_sequence, seq_mask=None): # seq_mask is the mask of what parts of generated_sequence are valid, 0 for valid, 1 for invalid, shape batch_size x target x target
        features = self.efficientnet(input_img)
        features = self.linear(features)
        features = self.activation(features)
        features = self.linear2(features)
        features = self.activation(features)
        features = torch.reshape(features, (1, -1, self.d_model))
        output_length = generated_sequence.shape[0]
        causal_mask = torch.triu(torch.ones(output_length, output_length, dtype=torch.bool), diagonal=1).to(features.device) # disallowed attention is 1, allow i to attend to j
        causal_mask = causal_mask.expand(input_img.shape[0] * self.nhead, -1, -1) # shape is batch_size * n heads x output_length x output_length
        # if seq_mask is None:
        #     combined_mask = causal_mask
        # else:
        #     processed_seq_mask = seq_mask.repeat_interleave(repeats=self.nhead, dim=0)
        #     combined_mask = torch.maximum(causal_mask, processed_seq_mask) # if either seq_mask or causal_mask disallow attention, then we disallow
        output = self.transformer_decoder(generated_sequence, features, tgt_mask=causal_mask, tgt_key_padding_mask=seq_mask) # > 0 cast to BoolTensor
        return output

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ChessRecognizerFull(torch.nn.Module):
    def __init__(self, chr2idx: Dict[str, int], recognizer_module: ChessRecognizer, dropout: float=0.5, device=torch.device('cpu')):
        super().__init__()
        self.recognizer = recognizer_module
        self.embedding_size = max(v for v in chr2idx.values()) + 1
        self.chr2idx = dict(chr2idx)
        self.idx2chr = dict((v, k) for k, v in self.chr2idx.items())
        self.embedding = torch.nn.Embedding(self.embedding_size, recognizer_module.d_model)
        self.pos_encoder = PositionalEncoding(recognizer_module.d_model, dropout)
        self.device = device
    
    def convert_text_to_idx_list(self, tokens_list: List[str], max_len: int) -> torch.Tensor:
        return torch.tensor([[self.chr2idx[tokens[i]] if i < len(tokens) else 0 for i in range(max_len)] for tokens in tokens_list], dtype=torch.long, device=self.device)
        
    def convert_text_to_tensor(self, tokens_list: List[str], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]: # returns a tuple of the embeddings and the mask
        # Remember to add start and end tokens
        # embedded is seq_length x d_model
        idx_tensor = self.convert_text_to_idx_list(tokens_list, max_len)
        embedded = self.embedding(idx_tensor) # comes out as batch x seq_length x d_model
        # now it is seq_length x batch x d_model, which is the convention for Transformer in PyTorch
        embedded = torch.transpose(embedded, 0, 1)
        embedded *= math.sqrt(self.recognizer.d_model)
        embedded = self.pos_encoder(embedded)
        
        seq_mask = torch.zeros((len(tokens_list), max_len), dtype=torch.bool, device=self.device)
        for item_number, tokens in enumerate(tokens_list):
            seq_mask[item_number, len(tokens):] = True # mask out invalid tokens
        return embedded, seq_mask
    
    def convert_output_tensor_to_logits(self, output_tensor):
        return output_tensor @ self.embedding.weight.t()
    
    def generate_output(self, input_img, max_len: int=90) -> str:
        with torch.no_grad():
            current_seq = self.idx2chr[1]
            curr_idx = 0
            while len(current_seq) < max_len:
                current_seq_tensor, curr_seq_mask = self.convert_text_to_tensor([current_seq], max_len)
                recognizer_output = self.recognizer(torch.unsqueeze(input_img, 0), current_seq_tensor, curr_seq_mask) # comes out as max_len x 1 x d_model
                recognizer_output = self.convert_output_tensor_to_logits(recognizer_output)
                pre_logits = recognizer_output[-1, :, :] # take the last element of the sequence, now it is 1 x d_model
                filtered_logits = top_k_top_p_filtering(pre_logits, top_k=0, top_p=0.85) # disable top_k, set top_p = 0.85
                filtered_logits = torch.nn.functional.softmax(pre_logits, dim=1) # still 1 x d_model
                next_token = torch.multinomial(filtered_logits, 1, replacement=True).item()
                if next_token == 0:
                    break
                current_seq += self.idx2chr[next_token]
            return current_seq[1:]
    
    def forward(self, input_img, desired_output_texts: List[str], max_len: int=90):
        shifted_outputs = [(self.idx2chr[2] + output)[:-1] for output in desired_output_texts] # Add a start token, corresponding to index 1 and clip off the last token, which is to be predicted
        current_seq_tensor, curr_seq_mask = self.convert_text_to_tensor(shifted_outputs, max_len)
        recognizer_output = self.recognizer(input_img, current_seq_tensor, curr_seq_mask)
        recognizer_output = self.convert_output_tensor_to_logits(recognizer_output)
        return recognizer_output, curr_seq_mask
    
    def compute_loss(self, input_img, desired_output_texts: List[str], max_len: int=90):
        desired_output_texts = [txt + self.idx2chr[1] for txt in desired_output_texts]
        # recognizer_output comes out as max_len x batch_size x vocab size, mask comes out as batch_size x max_len x max_len
        recognizer_output, mask = self(input_img, desired_output_texts, max_len)
        idx_list = self.convert_text_to_idx_list(desired_output_texts, max_len) # comes out as batch_size x max_len
        loss = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)(torch.permute(recognizer_output, (1, 2, 0)), idx_list)
        return torch.mean(loss)
    
    def compute_accuracy(self, input_img, desired_output_texts: List[str], max_len: int=90):
        with torch.no_grad():
            desired_output_texts = [txt + self.idx2chr[0] for txt in desired_output_texts]
            # recognizer_output comes out as max_len x batch_size x vocab size, mask comes out as batch_size x max_len x max_len
            recognizer_output, mask = self(input_img, desired_output_texts, max_len)
            recognizer_output = recognizer_output.detach()
            mask = mask.detach()
            # Index at 0 for the second coordinate, because it's just repeated along that dimension
            mask = 1 - mask.float()
            idx_list = self.convert_text_to_idx_list(desired_output_texts, max_len) # comes out as batch_size x max_len
            predicted_outputs = torch.permute(torch.argmax(recognizer_output, -1), (1, 0)) # batch_size x max_len
            differences = (idx_list != predicted_outputs) & (mask > 0.001) # should be False if correct
            rowwise_correct = torch.sum(differences, axis=-1) == 0
            rowwise_accuracy = torch.sum(rowwise_correct) / rowwise_correct.shape[0]
            if rowwise_accuracy > 0:
                correct_idx = torch.argmax(rowwise_correct).item()
                print(desired_output_texts[correct_idx])
                print(predicted_outputs[correct_idx])

            all_accuracy = 1 - torch.sum(differences) / torch.sum(mask)
            return rowwise_accuracy.item(), all_accuracy.item(), torch.sum(mask).item(), rowwise_correct.shape[0]

def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    import torch.nn.functional as F
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_p = float(top_p)
    if top_k > 0:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if 0 < top_p <= 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits