"""Evaluate CSV columns ground_truth_m,predicted_m on a held-out measurement set."""
import argparse
import csv
import json
import math


def evaluate(rows):
    pairs = [(float(r['ground_truth_m']), float(r['predicted_m'])) for r in rows]
    if not pairs or any(not math.isfinite(v) or v <= 0 for pair in pairs for v in pair):
        raise ValueError('Measurements must be nonempty, finite and positive')
    n = len(pairs)
    return {'n': n,
            'mae_m': sum(abs(p - g) for g, p in pairs) / n,
            'rmse_m': math.sqrt(sum((p - g) ** 2 for g, p in pairs) / n),
            'abs_rel': sum(abs(p - g) / g for g, p in pairs) / n,
            'delta_1': sum(max(p / g, g / p) < 1.25 for g, p in pairs) / n,
            'overestimate_gt_0_5m': sum(p - g > .5 for g, p in pairs) / n}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path')
    args = parser.parse_args()
    with open(args.csv_path, newline='') as stream:
        print(json.dumps(evaluate(csv.DictReader(stream)), indent=2))
