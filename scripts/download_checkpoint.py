import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve

from dotenv import load_dotenv


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("CHECKPOINT_URL"))
    parser.add_argument("--output", default="checkpoints/model_best.pth")
    args = parser.parse_args()
    if args.url is None:
        print("передайте --url или задайте CHECKPOINT_URL в .env")
        return
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(args.url, output)
    print(output)


if __name__ == "__main__":
    main()
