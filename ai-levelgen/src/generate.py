import argparse, pickle, random, torch, torch.nn as nn

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

def generate_lstm(ckpt_path, length=1000, temperature=0.8, seed_txt="---"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    stoi, itos = ckpt["stoi"], ckpt["itos"]
    model = CharLSTM(len(stoi))
    model.load_state_dict(ckpt["model"])
    model.eval()
    if not seed_txt:
        seed_txt = random.choice(list(stoi.keys()))

    idx = torch.tensor([[stoi.get(c, 0) for c in seed_txt]], dtype=torch.long)
    out_chars = list(seed_txt)
    h = None
    with torch.no_grad():
        for _ in range(length):
            logits, h = model(idx, h)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1).squeeze()
            next_id = torch.multinomial(probs, num_samples=1).item()
            out_chars.append(itos[next_id])
            idx = torch.tensor([[next_id]], dtype=torch.long)
    return "".join(out_chars)

def generate_ngram(ckpt_path, length=1000, seed="---"):
    ckpt = pickle.load(open(ckpt_path, "rb"))
    model, n = ckpt["model"], ckpt["n"]
    ctx = seed[:n-1] if len(seed) >= n-1 else seed + "-"*(n-1-len(seed))
    out = list(ctx)
    import random as _r
    for _ in range(length):
        dist = model.get(ctx)
        if not dist:
            ctx = _r.choice(list(model.keys()))
            dist = model[ctx]
        chars, probs = zip(*dist.items())
        r = _r.random()
        cumul = 0.0
        for c,p in zip(chars, probs):
            cumul += p
            if r <= cumul:
                out.append(c); break
        ctx = "".join(out[-(n-1):])
    return "".join(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm","ngram"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--length", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.model == "lstm":
        text = generate_lstm(args.ckpt, length=args.length)
    else:
        text = generate_ngram(args.ckpt, length=args.length)
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(text)
    print(f"Wrote generated sample to {args.out}")
