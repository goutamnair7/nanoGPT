import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
block_size = 8 # context length
batch_size = 32 # batches
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
eval_interval = 300
train_iters = 9000
learning_rate = 1e-3

# read input
with open('input.txt', 'r') as f:
    text = f.read()

# prepare input and encode chars to int
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
eval_data = data[n:]

def get_batch(split):
    split_data = train_data if split == 'train' else eval_data
    random_idx = torch.randint(len(split_data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in random_idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in random_idx])
    x, y = x.to(device), y.to(device)
    return x, y

# estimate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# define the model
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx = batch x block_size
        # logits = batch x block_size x vocab_size
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            logits = logits.view(batch_size*block_size, vocab_size)
            targets = targets.view(batch_size*block_size)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            last_step_logits = logits[:,-1,:]
            probs = F.softmax(last_step_logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx,  next_idx), dim=1)
        return idx

model = BigramModel(vocab_size)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(train_iters):
    if iter%eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['eval']:.4f}")
    
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

input = torch.zeros((1,1), dtype=torch.long, device=device)
generated_idx = model.generate(input, max_new_tokens=500)
print(decode(generated_idx[0].tolist()))