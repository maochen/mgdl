import csv
import math
import os
import sys
import tempfile
import random

from util.utils import get_idx_from_header


def get_bucket(score, min_val, max_val, buckets: int):  # return 0 to buckets-1
    portion = buckets / max_val
    if score == 0:
        score += 0.0000000000001

    bucket = math.ceil(portion * (score - min_val))
    return int(bucket) - 1


def get_bucket_histogram(input_file: str, total_buckets: int = 10):
    with open(input_file, "r", encoding="utf-8") as f:
        f_csv = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)

        header = next(f_csv)

        request_header = ["id", "gt", "score"]
        cols = get_idx_from_header(header, request_header)

        min_score = sys.maxsize
        max_score = 0
        data = {}

        for r in f_csv:
            if len(r) == 0:
                continue

            id = r[cols[0]].strip()
            gt_val = int(r[cols[1]].strip())  # 1 - Positive, 0 - NEG
            score = float(r[cols[2]])

            min_score = min(score, min_score)
            max_score = max(score, max_score)
            data[id] = (score, gt_val)

        buckets = [[0, 0] for _ in range(total_buckets)]  # True vs total

        for id, val in data.items():
            score = val[0]
            gt_val = val[1]
            bucket = get_bucket(score, min_score, max_score, total_buckets)

            if gt_val == 1:
                buckets[bucket][0] += 1

            buckets[bucket][1] += 1

        acc_list = [0] * len(buckets)
        for i, v in enumerate(buckets):
            if v[1] == 0:
                acc_list[i] = 0
            else:
                acc_list[i] = float(v[0]) / v[1]

        print("Bucket\tPOS Count\tTotal Count")
        for idx, tp in enumerate(buckets):
            print(f"{idx + 1}\t{tp[0]}\t{tp[1]}")
        print("==============================")
        print("Acc list")
        for idx, acc in enumerate(acc_list):
            print(f"{idx + 1}\t{acc}")


if __name__ == "__main__":
    data = []
    data.append("id\tgt\tscore\n")

    for i in range(100):
        gt = 1 if random.random() > 0.5 else 0
        score = random.random()
        data.append(f"{i + 1}\t{gt}\t{score}\n")

    fn = os.path.join(tempfile.gettempdir(), "test.tsv")
    with open(fn, "w") as f:
        f.writelines(data)

    get_bucket_histogram(fn)
