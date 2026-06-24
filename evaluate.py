"""
evaluate.py
  python evaluate.py --dataset hm --feature clip         --model lightgcn
  python evaluate.py --dataset hm --feature clip         --model graphsage
  python evaluate.py --dataset hm --feature clip         --model ngcf
  python evaluate.py --dataset hm --feature fashionclip  --model lightgcn
  python evaluate.py --dataset hm                        --model bpr
  python evaluate.py --dataset hm --feature clip         --model siamese
  python evaluate.py --dataset hm
"""

import argparse
from src.evaluation.runner import run_evaluation


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature", default=None)
    parser.add_argument("--model",   default=None)
    args = parser.parse_args()

    run_evaluation(
        dataset    = args.dataset,
        feature    = args.feature,
        model_name = args.model,    
    )


if __name__ == "__main__":
    main()