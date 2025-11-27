import argparse, os, glob

def read_levels(input_dir):
    texts = []
    for fp in glob.glob(os.path.join(input_dir, "*.txt")):
        with open(fp, "r", encoding="utf-8") as f:
            texts.append(f.read().strip() + "\n\n")  # separator between levels
    return "".join(texts)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    text = read_levels(args.input_dir)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote corpus of {len(text)} characters to {args.out}")
