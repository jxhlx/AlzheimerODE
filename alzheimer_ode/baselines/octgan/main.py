import os
import sys
import warnings
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / 'octgan'))

from octgan.benchmark import run
from octgan.synthesizers.octgan import OCTGANSynthesizer as Synthesizer

warnings.filterwarnings(action='ignore')

def main():
    parser = argparse.ArgumentParser("ctgan with odes")
    parser.add_argument('--dataset_name', type=str, default='ATN')
    parser.add_argument('--synthesizer', type=str, default='octgan')
    parser.add_argument('--gen_dim', nargs='+', type=int, default=(128, 128))
    parser.add_argument('--dis_dim', nargs='+', type=int, default=(128, 128))
    parser.add_argument('--num_split', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--random_dim', type=int, default=128)
    parser.add_argument('--num_channels', type=int, default=64)
    parser.add_argument('--l2scale', type=float, default=1e-06)

    parser.add_argument('--lr', type=float, default=2e-3)

    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default=str(ROOT_DIR / 'outputs'))

    config = parser.parse_args()
    scores = run(Synthesizer, arguments=config, output_path=config.output_dir)
    print(scores)


if __name__ == '__main__':
    main()
