import argparse
from ranx import Qrels, Run
from ranx import fuse


def read_trec_run(file):
    run = {}
    with open(file, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if qid not in run:
                run[qid] = {}
            run[qid][docid] = float(score)
    return run


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_1", type=str)
    parser.add_argument("--run_2", type=str)
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--save_path", type=str)

    args = parser.parse_args()

    run1 = read_trec_run(args.run_1)
    run2 = read_trec_run(args.run_2)

    # handle queries that are not in both runs
    qids = set(run1.keys()).union(set(run2.keys()))
    for qid in qids:
        if qid not in run1:
            run1[qid] = run2[qid]
        if qid not in run2:
            run2[qid] = run1[qid]

    print('fusing runs')
    fusion_run = fuse(
        runs=[Run.from_dict(run1),
              Run.from_dict(run2)],
        norm="min-max",
        method="wsum",
        params={"weights": [args.alpha, (1-args.alpha)]}

    )

    fusion_run.save(args.save_path, kind='trec')
