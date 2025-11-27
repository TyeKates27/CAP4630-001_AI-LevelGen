import argparse, torch, torch.nn as nn, torch.optim as optim, os

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=128, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)
    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.lstm(x, h)
        logits = self.fc(out)
        return logits, h

def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {c:i for i,c in enumerate(chars)}
    itos = {i:c for c,i in stoi.items()}
    return stoi, itos

def batchify(seq, seq_len=128, step=128):
    X, Y = [], []
    for i in range(0, len(seq)-seq_len-1, step):
        X.append(seq[i:i+seq_len])
        Y.append(seq[i+1:i+seq_len+1])
    return torch.tensor(X), torch.tensor(Y)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = open(args.data, "r", encoding="utf-8").read()
    stoi, itos = build_vocab(text)
    encoded = [stoi[c] for c in text]
    X, Y = batchify(encoded, seq_len=args.seq_len, step=args.seq_len)
    model = CharLSTM(len(stoi)).to(device)
    opt = optim.Adam(model.parameters(), lr=3e-3)
    crit = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total = 0
        for i in range(len(X)):
            xb = X[i].unsqueeze(0).to(device)
            yb = Y[i].unsqueeze(0).to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = crit(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch+1}: loss={total/len(X):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model":model.state_dict(), "stoi":stoi, "itos":itos}, args.out)
    print(f"Saved LSTM to {args.out}")
