import argparse
import pandas as pd
from tqdm import tqdm

from utils import split

def main():
    parser = argparse.ArgumentParser(description='Build a corpus file')
    parser.add_argument('--in_path', '-i', type=str, default='../AID_CSV/AID.csv', help='input file')
    parser.add_argument('--out_path', '-o', type=str, default='data/AID_corpus.txt', help='output file')
    args = parser.parse_args()

    try:
        smiles = pd.read_csv(args.in_path)['SMILES'].values
    except:
        smiles = pd.read_csv(args.in_path)['canonical_smiles'].values
    with open(args.out_path, 'a') as f:
        for sm in tqdm(smiles):   
            f.write(split(sm)+'\n')
    print('Built a corpus file!')

if __name__=='__main__':
    main()



