import argparse, pickle, collections, random, os

def train_ngram(corpus, n=3):
    counts = collections.defaultdict(collections.Counter)
    for i in range(len(corpus)-n):
        ctx = corpus[i:i+n-1]
        nxt = corpus[i+n-1]
        counts[ctx][nxt] += 1
    # add-1 smoothing to build probs
    model = {}
    for ctx, ctr in counts.items():
        total = sum(ctr.values()) + len(ctr)
        model[ctx] = {ch:(count+1)/total for ch,count in ctr.items()}
    return model

def sample(model, length=1000, seed="---"):
    ctx = seed
    out = list(ctx)
    for _ in range(length - len(ctx)):
        dist = model.get(ctx)
        if not dist:
            # random restart
            ctx = random.choice(list(model.keys()))
            dist = model[ctx]
        # sample next char
        chars, probs = zip(*dist.items())
        import random as _r
        r = _r.random()
        cumul = 0.0
        for c,p in zip(chars, probs):
            cumul += p
            if r <= cumul:
                out.append(c); break
        ctx = "".join(out[-(len(ctx)):])
    return "".join(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.data, "r", encoding="utf-8") as f:
        corpus = f.read()
    model = train_ngram(corpus, n=args.n)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"n":args.n, "model":model}, f)
    print(f"Saved n-gram model (n={args.n}) to {args.out}")
