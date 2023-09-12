import numpy as np
import torch
from torch import nn
import torch.optim as optim


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()
        vocab_size = 500
        embedding_dim = 10
        hidden_size = 2
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.rnn = nn.GRU(embedding_dim, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.m = nn.Sigmoid()

    def forward(self, batch_ind):
        # batch: batch_size x seq_length
        x = self.embedding(batch_ind)
        x, _ = self.rnn(x)
        x = self.out(x)
        return self.m(x[-1, :, :].squeeze())

def main():
    with open('X.npy', 'rb') as i_f:
        X = np.load(i_f)
    X = torch.from_numpy(X)
    with open('y.npy', 'rb') as i_f:
        y = np.load(i_f)
    y = torch.from_numpy(y.astype(np.float32))

    model = MyRNN()
    num_params = count_parameters(model)
    print(f'Number of trainable parameters: {num_params}')
    __import__('pdb').set_trace()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    shuffled_ind = np.random.permutation(X.shape[1])
    batch_size = 99
    num_epochs = 1000
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
