import io
import math

import PIL
import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
from sklearn.metrics import average_precision_score
from torchvision.transforms import ToTensor


def __autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.0 * height,  # y offset
                "%.2f" % height,
                ha='center', va='bottom')


def __add_label_distribution_binary(gts, y_probs, tbwriter, tb_tag: str, global_step: int = 0):
    """
    :param gts: Groundtruth label int indices. shape(n_total_samples,)
    :param y_probs: same shape as gt_labels. shape(n_total_samples, )
    :param tbwriter: tensorboard writer.
    :param tb_tag: tensorboard writer tag.
    :param global_step: int tensorboard global step.
    """

    label_prob = {}
    if len(gts) != len(y_probs):
        raise ValueError(f"gt_label size [{len(gts)}] and prob size [{len(y_probs)}] mismatch.")

    for idx, label in enumerate(gts):
        arr = label_prob.get(label, [])
        arr.append(y_probs[idx])
        label_prob[label] = arr

    plt.ioff()
    fig, ax = plt.subplots()

    color = ["r", "g", "b"]
    idx = 0
    for k in sorted(label_prob.keys()):
        v = label_prob[k]
        ax.hist(v, color=color[idx % len(color)], alpha=0.5, label=k)
        fig.gca().set(title="Prob Histogram", xlabel="Confidence", ylabel="Freq")
        fig.legend()
        idx += 1

    tbwriter.add_figure(tb_tag, fig, global_step=global_step)


def __add_calibration_label_dist_binary(gts, y_probs, tbwriter, tb_tag: str, buckets: int = 10, global_step: int = 0):
    if len(gts) != len(y_probs):
        raise ValueError(f"gt_label size [{len(gts)}] and prob size [{len(y_probs)}] mismatch.")

    y_buckets = [[0, 0] for _ in range(buckets)]  # [pos #, total #]

    for idx, label in enumerate(gts):
        y_bucket = y_probs[idx] * buckets / float(1.0)  # MAX VALUE
        y_bucket = math.floor(y_bucket)

        if y_bucket >= buckets:  # fix the last one to the prev.
            y_bucket -= 1

        pos, total = y_buckets[y_bucket]

        if label == 1:
            pos += 1

        y_buckets[y_bucket] = [pos, total + 1]

    y_buckets = [(float(y[0]) / y[1]) if y[1] != 0 else 0 for y in y_buckets]

    plt.ioff()
    fig, ax = plt.subplots()

    rects = ax.bar(x=[i for i in range(buckets)], height=y_buckets)
    __autolabel(rects, ax)

    fig.gca().set(title="Positive Rate", xlabel="Buckets", ylabel="Percentage")
    fig.legend()

    tbwriter.add_figure(tb_tag, fig, global_step=global_step)


def __add_roc(gts, y_prob, idx_to_label: {}, tbwriter, tb_tag="roc", global_step: int = 0):
    """
    :param gts: Groundtruth label int indices. shape(n_total_samples, 1)
    :param y_prob: shape(n_total_samples, n_classes)
    :param tbwriter: Tensorboard SummaryWriter
    :param tb_tag: Optional. Tag for the plot in tbwriter
    :return: None
    """

    plt.ioff()
    skplt.metrics.plot_roc(gts, y_prob, plot_macro=False, plot_micro=False, title="ROC " + str(idx_to_label),
                           title_fontsize="medium")
    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches='tight', format='png')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    tbwriter.add_image(tb_tag, image[0], global_step=global_step)
    plt.clf()


def __add_pr(gts, y_prob, tbwriter, tb_tag="PR_Curve", global_step: int = 0):
    """
    :param gts: Groundtruth label int indices. shape(n_total_samples, 1)
    :param y_prob: shape(n_total_samples, 1)
    :param tbwriter: Tensorboard SummaryWriter
    :param tb_tag: Optional. Tag for the plot in tbwriter
    :return: None
    """

    plt.ioff()
    skplt.metrics.plot_precision_recall(gts, y_prob, plot_micro=True)

    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches='tight', format='png')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    tbwriter.add_image(tb_tag, image[0], global_step=global_step)
    plt.clf()


def add_all_metrics(tboard_writer, gts, y_preds, label2idx: {}, tag: str, train_ep: int = 0):
    """
    Binary Class Metrics ONLY FOR NOW.

    :param tboard_writer: TensorBoard SummaryWriter
    :param gts: GroundTruth array. shape(n_samples)
    :param y_preds: Prediction Probability. shape(n_samples)  // shape(n_samples, n_classes)
    :param label2idx: label to index.
    :param tag: tag for TensorBoard
    :param train_ep: int. (Optional) train epoch.
    """
    gts = np.asarray(gts)
    y_preds = np.asarray(y_preds)
    idx2label = dict([(value, key) for key, value in label2idx.items()])

    label_count = {}
    for y_idx in gts:
        curr_count = label_count.get(idx2label[y_idx], 0) + 1
        label_count[idx2label[y_idx]] = curr_count

    tboard_writer.add_text("meta", f"Total count: {len(gts)}  \nLabel count: {label_count}", global_step=train_ep)

    # TODO: ONLY BINARY, need to change to multi-class
    if y_preds.ndim == 1:
        y_pred_binary = y_preds
    else:
        y_pred_binary = [x[1] if gts[idx] == 1 else x[0] for idx, x in enumerate(y_preds)]

    __add_label_distribution_binary(gts, y_pred_binary, tboard_writer, tag + "/label_dist", train_ep)
    __add_calibration_label_dist_binary(gts, y_pred_binary, tboard_writer, tag + "/label_dist_calibrated",
                                        global_step=train_ep)
    average_precision = average_precision_score(gts, y_pred_binary)
    tboard_writer.add_text(tag, f"Average precision: {average_precision}", global_step=train_ep)

    # for idx in range(y_preds.shape[1]):
    #     curr_y_pred = y_preds[:, idx:idx + 1]
    #     curr_y = torch.Tensor([1 if x == idx else 0 for x in gts]).unsqueeze(1)
    #
    #     tag_suffix = [k for (k, v) in label2idx.items() if v == idx][0]
    #     tboard_writer.add_pr_curve(tag + "/" + tag_suffix, curr_y, curr_y_pred)

    tag_suffix = idx2label[1]
    tboard_writer.add_pr_curve(str(tag) + "/" + str(tag_suffix), gts, y_preds)

    # =================

    y_preds_multiple = [[1 - i, i] for i in y_preds]
    __add_roc(gts, y_preds_multiple, idx2label, tboard_writer, tag + "/roc", train_ep)
    __add_pr(gts, y_preds_multiple, tboard_writer, tag + "/pr", train_ep)
