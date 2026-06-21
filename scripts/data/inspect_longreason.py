import argparse
import json
from datasets import load_dataset, get_dataset_split_names

def short(x, n=500):
    s = str(x)
    return s[:n] + ("..." if len(s) > n else "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lz1bytedance/LongReason")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--streaming", action="store_true")
    args = ap.parse_args()

    print("Dataset:", args.dataset)

    try:
        splits = get_dataset_split_names(args.dataset)
        print("Available splits:", splits)
    except Exception as e:
        print("Could not get split names:", repr(e))
        splits = args.splits or ["original", "expanded", "8k"]

    if args.splits:
        splits = args.splits

    for sp in splits:
        print("\n" + "=" * 100)
        print("SPLIT:", sp)
        try:
            ds = load_dataset(args.dataset, split=sp, streaming=args.streaming)
            if args.streaming:
                it = iter(ds)
                row = next(it)
                print("streaming: yes")
                print("length: unknown in streaming mode")
            else:
                print("length:", len(ds))
                print("columns:", ds.column_names)
                row = ds[0]

            print("first row keys:", list(row.keys()))
            print("first row preview:")
            for k, v in row.items():
                print(f"- {k}: {short(v)}")

        except Exception as e:
            print("FAILED:", repr(e))

if __name__ == "__main__":
    main()
