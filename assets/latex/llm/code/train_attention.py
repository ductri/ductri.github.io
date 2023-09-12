import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / np.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MyAttention(nn.Module):
    def __init__(self):
        super(MyAttention, self).__init__()
        vocab_size = 60
        embedding_dim = 10
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.proj = nn.Linear(embedding_dim, 3*embedding_dim)

        self.out = nn.Linear(embedding_dim, 1)
        self.m = nn.Sigmoid()

    def forward(self, batch_ind):
        # batch: batch_size x seq_length
        x = self.embedding(batch_ind)
        x = self.proj(x)
        batch_size = x.shape[1]
        seq_length = x.shape[0]
        q, k, v = x.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v)
        values = values.mean(0)
        x = self.out(values)
        return self.m(x.squeeze())

def main():
    with open('X.npy', 'rb') as i_f:
        X = np.load(i_f)
    X = torch.from_numpy(X)
    with open('y.npy', 'rb') as i_f:
        y = np.load(i_f)
    y = torch.from_numpy(y.astype(np.float32))

    model = MyAttention()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    shuffled_ind = np.random.permutation(X.shape[1])
    batch_size = 100
    num_epochs = 2000
    for epoch in range(num_epochs):
        for i in range(X.shape[1]//batch_size-1):
            inputs = X[:, shuffled_ind[batch_size*i:batch_size*(i+1)]]
            labels = y[shuffled_ind[batch_size*i:batch_size*(i+1)]]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = (((outputs>=0.5) == labels)*1.0).mean()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f} Acc: {acc:.3f}')


if __name__ == "__main__":
    main()
