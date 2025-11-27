import argparse, math, collections

def ngram_counts(text, n=3):
    counts = collections.Counter()
    for i in range(len(text)-n+1):
        counts[text[i:i+n]] += 1
    total = sum(counts.values())
    return counts, total

def kl_divergence(p_counts, p_total, q_counts, q_total, n=3, eps=1e-8):
    # add-1 smoothing
    vocab = set(p_counts.keys()) | set(q_counts.keys())
    kl = 0.0
    for g in vocab:
        p = (p_counts.get(g,0)+1)/(p_total+len(vocab))
        q = (q_counts.get(g,0)+1)/(q_total+len(vocab))
        kl += p * math.log((p+eps)/(q+eps))
    return kl

def simple_rule_playability(text, tile_set, width):
    # Check rows are same width and there is at least one solid ground 'X' per column.
    rows = [r for r in text.splitlines() if r.strip()!=""]
    ok_width = all(len(r)==width for r in rows)
    if not ok_width: return False, "inconsistent width"
    for col in range(width):
        col_tiles = [rows[r][col] for r in range(len(rows))]
        if 'X' not in col_tiles:
            return False, f"no ground in column {col}"
    whitelist = set(tile_set)
    for ch in "".join(rows):
        if ch not in whitelist:
            return False, f"unknown tile '{ch}'"
    return True, "pass"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference corpus")
    ap.add_argument("--gen", required=True, help="generated sample")
    ap.add_argument("--tile_set", default="X-?E")
    ap.add_argument("--width", type=int, default=136)
    args = ap.parse_args()

    ref = open(args.ref, "r", encoding="utf-8").read()
    gen = open(args.gen, "r", encoding="utf-8").read()

    p_counts, p_total = ngram_counts(ref, n=3)
    q_counts, q_total = ngram_counts(gen, n=3)
    kl = kl_divergence(p_counts, p_total, q_counts, q_total, n=3)
    playable, reason = simple_rule_playability(gen, args.tile_set, width=args.width)

    print({"kl_3gram": round(kl, 4), "playable": playable, "reason": reason})
