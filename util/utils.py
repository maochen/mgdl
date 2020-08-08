import base64
import binascii
import collections
import csv
import logging
import os
import re
import sys
import tempfile

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, WeightedRandomSampler, SubsetRandomSampler

logger = logging.getLogger(__name__)


class NoReturnLoggerFilter(logging.Filter):
    def filter(self, record):
        return False


def batch_padding(vector_arr: [], padding_value: int, right_padding: bool = True,
                  max_padding_len: int = sys.maxsize) -> (torch.Tensor, torch.Tensor):
    """

    :param vector_arr: list of tensors
    :param padding_value:
    :param right_padding:
    :param max_padding_len:
    :return:
    """
    maxlen = max([x.shape[0] for x in vector_arr])
    maxlen = min(maxlen, max_padding_len)

    if right_padding:
        padded = np.array([i.tolist() + [padding_value] * (maxlen - len(i)) for i in vector_arr])
    else:
        padded = np.array([[padding_value] * (maxlen - len(i)) + i.tolist() for i in vector_arr])

    attn_mask = np.where(padded != 0, 1, 0)

    return torch.tensor(padded, dtype=torch.int64), torch.tensor(attn_mask, dtype=torch.int64)


def fix_incorrect_padding(data: str, altchars=b'+/'):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    data = re.sub(r'[^a-zA-Z0-9%s]+' % altchars, '', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += '=' * (4 - missing_padding)
    return base64.b64decode(data, altchars)


def get_idx_from_header(header: [], header_names: []):
    header = [x.lower() for x in header]
    header_names = [x.lower() for x in header_names]

    result = [-1] * len(header_names)
    for idx, col in enumerate(header):
        if col in header_names:
            j = header_names.index(col)
            result[j] = idx

    return result


def get_sampler_4_balanced_classes(dataset: Dataset, num_classes: int, prev_selected_indices: list = None):
    """
    For using this method, please implement dataset.get_label_only(idx) that returns the label index method.
    Otherwise, it will call __get_item__(), slow.

    :param dataset: dataset.
    :param num_classes: label count
    :param prev_selected_indices: Optional. If Previous has another sampler that select out some indices.
    :return:
    """

    if not prev_selected_indices:
        prev_selected_indices = [i for i in range(len(dataset))]

    count = [0] * num_classes

    idx_to_label = {}

    for idx in prev_selected_indices:
        if hasattr(dataset, "get_label_only"):
            curr_label = dataset.get_label_only(idx)
        else:  # This could be slow as it calls __get_item__()
            item = dataset[idx]
            curr_label = item[1]

        count[curr_label] += 1
        idx_to_label[idx] = curr_label

    weight_per_class = [0.] * num_classes
    total_count = float(sum(count))
    for i in range(num_classes):
        if count[i] == 0:
            weight_per_class[i] = 0
        else:
            weight_per_class[i] = total_count / float(count[i])

    weight = [0.] * len(dataset)
    for idx in prev_selected_indices:
        weight[idx] = weight_per_class[idx_to_label[idx]]

    max_class_count = max([v for k, v in collections.Counter([x for x in weight if x != 0]).items()])
    # Make max class instance*1.6 size to upsampling not allowed classes.
    sampler = WeightedRandomSampler(weight, int(max_class_count * num_classes * 0.8), replacement=True)
    return sampler

