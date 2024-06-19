import time
from tqdm import tqdm
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=float, default=0)
    parser.add_argument('--subbase', type=float, default=1)
    parser.add_argument('--iter', type=float, default=0)
    args = parser.parse_args()
    for i in tqdm(range(int(args.base + args.subbase * args.iter))):
        time.sleep(60)
