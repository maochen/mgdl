import io
import logging
import os
import sys
import time
import traceback
from abc import abstractmethod
from collections import OrderedDict
from subprocess import call

import numpy as np
import torch
import tqdm

from torch import nn
from torch.backends import cudnn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from util.metrics import add_all_metrics

logger = logging.getLogger(__name__)


class AbstractClassifier:
    model: nn.Module = None  # Do NOT wrap to DATAPARALLEL() here, do all these when using this model.
    optimizer: Optimizer = None
    criterion = None
    label2idx = None
    input_shape = None
    tqdm_logger = None
    metadata = {}  # write all other states here, it will serialize and deserialize

    @abstractmethod
    def get_model(self, num_classes: int, **kwargs) -> (nn.Module, object, Optimizer):
        pass

    @abstractmethod
    def load_data(self, **kwargs) -> (DataLoader, {}):
        """
        please set self.label2idx

        :param kwargs:
        :return: Dataloader, label2idx dict
        """
        pass

    def __init__(self):
        class TqdmToLogger(io.StringIO):
            """
                Output stream for TQDM which will output to logger module instead of
                the StdOut.
            """
            logger = None
            level = None
            buf = ''

            def __init__(self, logger, level=None):
                super(TqdmToLogger, self).__init__()
                self.logger = logger
                self.level = level or logging.INFO

            def write(self, buf):
                self.buf = buf.strip('\r\n\t ')

            def flush(self):
                self.logger.log(self.level, self.buf)

        self.tqdm_logger = TqdmToLogger(logger, level=logging.INFO)

    def fit(self, train_dl: DataLoader, val_dl: DataLoader, epochs: int, saved_model_dir: str,
            model_id: str, tensorboard_dir: str = None, max_model_saved: int = 5):
        if not self.label2idx:
            raise ValueError("self.label2idx is None. Please set this in load_data()")

        if tensorboard_dir:
            tboard_writer = SummaryWriter(comment=model_id, filename_suffix="." + model_id, log_dir=tensorboard_dir)
            logger.info(f"Write metrics to tensorboard in {tensorboard_dir}")

            # input_tensors = []
            # for input_shape in self.model.input_shape:
            #     t = torch.zeros(input_shape[0], dtype=input_shape[1])
            #     input_tensors.append(t)
            #
            # tboard_writer.add_graph(self.model, input_to_model=input_tensors)
            # TODO: sth is wrong with this on multi GPU train.
        else:
            logger.info("No tensorboard dir specified, will skip metrics.")

        logger.info(f"Total train dataset size: {len(train_dl.dataset)} / Found labels [{self.label2idx}]")
        logger.info(f"Training Started ... [Max Model Saved: {max_model_saved}]")

        self.model, self.criterion = self.__auto_transformed_cuda_model(self.model, self.criterion)
        logger.info(f"Loss Fn: {type(self.criterion)}")
        saved_model_filenames = {}  # Key - ep, Value - filename

        for ep in range(epochs):
            total_train_count = 0

            # Train
            self.model.train()  # IMPORTANT
            running_loss, correct = 0.0, 0
            train_dl_tqdm = tqdm.tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {ep + 1}/{epochs}",
                                      file=self.tqdm_logger)
            try:
                for idx, (X, y, _) in train_dl_tqdm:  # use img path as row_id
                    if len(y) == 0:
                        continue
                    total_train_count += len(y)

                    if torch.cuda.is_available():
                        if type(X) is torch.Tensor:
                            X = X.cuda()
                        elif type(X) is list:
                            X = [t.cuda() for t in X]
                        else:
                            raise ValueError(f"Unrecognized X type: {type(X)}")

                        y = y.cuda()

                    self.optimizer.zero_grad()
                    y_pred = self.model(X)
                    _, y_pred_label_ = torch.max(y_pred, 1)
                    loss = self.criterion(y_pred, y)

                    loss.backward()
                    self.optimizer.step()

                    batch_correct = (y_pred_label_ == y).sum().item()
                    correct += batch_correct
                    running_loss += loss.item() * y.shape[0]

                    if tensorboard_dir:
                        tboard_writer.add_scalar("train/loss", running_loss / total_train_count,
                                                 global_step=ep * len(train_dl) + idx)
                        tboard_writer.add_scalar("train/acc", correct / total_train_count,
                                                 global_step=ep * len(train_dl) + idx)
                        tboard_writer.add_scalar("train/batch_loss", loss.item(), global_step=ep * len(train_dl) + idx)
                        tboard_writer.add_scalar("train/batch_acc", batch_correct / y.shape[0],
                                                 global_step=ep * len(train_dl) + idx)

                    train_dl_tqdm.set_postfix_str(
                        f"batch loss: {loss.item():0.6f} batch acc: {(batch_correct / y.shape[0]):0.6f}")
            except OSError:
                logger.warning(traceback.format_exc())

            loss_val = running_loss / total_train_count
            accuracy = correct / total_train_count
            metrics = {"loss": loss_val, "acc": accuracy}

            # Validation
            if val_dl:

                if tensorboard_dir:
                    detail_fn = os.path.join(tensorboard_dir, "train_err_prediction.tsv")
                else:
                    tboard_writer = None
                    detail_fn = None

                y_preds, val_metrics = self.predict(val_dl, tboard_writer, f"Validation", detail_fn=detail_fn,
                                                    train_epoch=ep)

                val_loss = val_metrics["loss"]
                val_accuracy = val_metrics["acc"]

                if tensorboard_dir:
                    tboard_writer.add_scalar("val/loss", val_loss, ep)
                    tboard_writer.add_scalar("val/acc", val_accuracy, ep)

                metrics["val_loss"] = val_loss
                metrics["val_acc"] = val_accuracy

            logger.info(
                f"Epoch {ep + 1}/{epochs} - loss: {loss_val:0.6f}  acc: {accuracy:0.6f} | val_loss: {metrics.get('val_loss', 0):0.6f} val_acc: {metrics.get('val_acc', 0):0.6f}")

            # Not saving optimizer
            filename = self.save_model(self.model, self.input_shape, self.label2idx, saved_model_dir,
                                       save_only_require_grad_layers=False, optimizer=None,
                                       metadata=self.metadata,
                                       model_id=model_id, ep=ep + 1, metrics=metrics)

            # Saved model ckpt housekeeping
            saved_model_filenames[ep] = filename
            model_ep_need_delete = max(-1, ep - max_model_saved)
            if model_ep_need_delete > -1:
                fn = saved_model_filenames[model_ep_need_delete]
                os.remove(fn)
                saved_model_filenames.pop(model_ep_need_delete, None)
            # ===================

        if tensorboard_dir:
            tboard_writer.close()

        logger.info(f"Training completed. Model saved to: {saved_model_dir}")

    def predict(self, predict_dl, tboard_writer: SummaryWriter = None, tboard_tag: str = None, detail_fn: str = None,
                train_epoch: int = -1):
        """

        :param predict_dl:
        :param tboard_writer:
        :param tboard_tag:
        :param detail_fn: output wrong prediction details to detail_fn.
        :param train_epoch: int. Optional. Default=-1 which is inference.
        :return:
        """
        if tboard_writer and not self.criterion:
            raise ValueError("write to tensorboard = True but missing criterion=None.")

        if not tboard_tag:
            tboard_tag = "predict_" + str(time.strftime('%Y%m%d%H%M%S', time.gmtime()))

        idx2label = dict([(value, key) for key, value in self.label2idx.items()])
        predict_ret = []
        y_pred_all = []
        y_all = []

        model, criterion = AbstractClassifier.__auto_transformed_cuda_model(self.model, self.criterion,
                                                                            show_log_info=train_epoch == 0)
        model.eval()  # IMPORTANT
        running_loss, correct, total_val_count, val_loss, val_accuracy = 0.0, 0, 0, 0, 0

        with torch.no_grad():  # IMPORTANT
            for batch_num, (x, y, row_id) in tqdm.tqdm(enumerate(predict_dl), total=len(predict_dl), desc="Evaluation",
                                                       file=self.tqdm_logger):
                if x is None:
                    continue

                if torch.cuda.is_available():
                    if type(x) is torch.Tensor:
                        x = x.cuda()
                    elif type(x) is list:
                        x = [t.cuda() for t in x]
                    else:
                        raise ValueError(f"Unrecognized X type: {type(x)}")

                    y = y.cuda()

                curr_pred = model(x)
                curr_pred_prob, curr_pred_label_ = torch.max(curr_pred, 1)

                y_pred_all.extend(curr_pred)
                y_all.extend(y)

                for idx, prob in enumerate(curr_pred_prob):
                    pred_row = [idx2label[curr_pred_label_[idx].item()], prob.item()]
                    if row_id:
                        pred_row.insert(0, row_id[idx])
                    else:
                        pred_row.insert(0, "PADDING_ID")

                    predict_ret.append(pred_row)

                if tboard_writer and train_epoch > -1:
                    total_val_count += y.shape[0]
                    loss = criterion(curr_pred, y)
                    correct += (curr_pred_label_ == y).sum().item()
                    running_loss += loss.item() * y.shape[0]

        if len(predict_ret) == 0:
            logger.warning("No predicting found.")
        elif tboard_writer and train_epoch > -1:
            val_loss = running_loss / (total_val_count + sys.float_info.min)
            val_accuracy = correct / (total_val_count + sys.float_info.min)
            tboard_writer.add_text(tboard_tag, f"loss: {val_loss} acc: {val_accuracy}", global_step=train_epoch)

            y_pred_all = torch.stack(y_pred_all).cpu().numpy()
            y_all = torch.stack(y_all).cpu().numpy()
            y_pred_all = y_pred_all[:, 1]  # TODO: After change add_all_metrics to multiclass, remove this.
            add_all_metrics(tboard_writer, y_all, y_pred_all, self.label2idx, tboard_tag, train_epoch)

            y_all_label = [idx2label[y] for y in y_all]
            AbstractClassifier.add_all_wrong_predictions(y_all_label, predict_ret, detail_fn, train_epoch)
            tboard_writer.add_text("detail file path", detail_fn)

        return predict_ret, {"loss": val_loss, "acc": val_accuracy}

    def load_model(self, model_file_path: str):
        if self.model is None:
            raise Exception("Please call get_model() to feed in model structure first.")

        if torch.cuda.is_available():
            model_dict = torch.load(model_file_path)
        else:
            model_dict = torch.load(model_file_path, map_location=torch.device("cpu"))

        self.input_shape = model_dict["input_shape"]

        model_state_dict = model_dict["model_state_dict"]
        AbstractClassifier.load_any_gpu_state_dict_to_model(self.model, model_state_dict)

        if "optimizer_state_dict" in model_dict:
            self.optimizer.load_state_dict(model_dict["optimizer_state_dict"])

        self.metadata = model_dict.get("metadata", None)

        self.label2idx = model_dict["labels"]

        return self.model, self.optimizer, self.criterion, self.label2idx, self.input_shape, self.metadata

    @staticmethod
    def save_model(model: nn.Module, input_shape: tuple, labels: {},
                   saved_model_dir: str, metadata: {} = None, optimizer: Optimizer = None, model_id: str = "",
                   save_only_require_grad_layers: bool = True, ep: int = None, metrics: {} = None):

        if save_only_require_grad_layers:
            saved_model_state_dict = OrderedDict()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    saved_model_state_dict[name] = model.state_dict()[name]
        else:
            saved_model_state_dict = model.state_dict()

        states = {
            "model_state_dict": saved_model_state_dict,
            "labels": labels,
            "input_shape": input_shape,
            "model_id": model_id
        }

        if optimizer:
            states["optimizer_state_dict"] = optimizer.state_dict()

        if metadata:
            states["metadata"] = metadata

        if ep and metrics:
            filename = os.path.join(saved_model_dir, f"model_{model_id}_ep{ep}_{metrics['acc']:0.3f}.pth")
        else:
            filename = os.path.join(saved_model_dir, f"model_{model_id}.pth")

        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)

        torch.save(states, filename)
        return filename

    @staticmethod
    def load_any_gpu_state_dict_to_model(model: nn.Module, state_dict: {}):
        """
        Load single GPU ckpt to nn.DataParallel model or vice versa depends on model type.

        :param model: padding model object that await load state_dict.
        :param state_dict: await converting model state_dict
        :return: None
        """
        if not model:
            raise ValueError("model is null.")
        elif not state_dict:
            raise ValueError("state_dict is null.")

        is_single_model = type(model) is not nn.DataParallel
        is_single_state_dict = not next(iter(state_dict)).startswith("module.")

        if is_single_model ^ is_single_state_dict:  # need convert
            new_dict = {}

            for k, v in state_dict.items():
                if is_single_state_dict:
                    new_dict["module." + k] = v
                else:
                    new_dict[k[7:]] = v  # remove `module.`
        else:
            new_dict = state_dict

        all_state_dict = model.state_dict()
        for k, v in new_dict.items():
            all_state_dict.update({k: v})

        model.load_state_dict(all_state_dict)

    @staticmethod
    def split_train_test(train_ds: Dataset, val_portion: float, from_idx: int = 0, to_idx: int = None) -> \
            (list, list, Dataset):
        if not to_idx:
            to_idx = len(train_ds)

        if val_portion == 1.0:
            logger.info("Eval on full training dataset.")
            indices = list(range(from_idx, to_idx))
            return indices, indices, train_ds

        indices = list(range(from_idx, to_idx))
        np.random.shuffle(indices)
        split = int(np.floor(val_portion * len(train_ds)))
        train_indices, val_indices = indices[split:], indices[:split]
        return train_indices, val_indices, train_ds

    @staticmethod
    def __auto_transformed_cuda_model(model, criterion, show_log_info: bool = True):
        if torch.cuda.is_available():
            if show_log_info:
                logger.info("Dispatch model on GPU ...")
                call(["nvidia-smi", "--format=csv",
                      "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])

            model = model.cuda()
            if criterion is not None:
                criterion = criterion.cuda()

            cudnn.benchmark = True
        elif show_log_info:
            logger.info("No CUDA detected! Dispatch model on CPU.")

        if type(model) is not nn.DataParallel:
            model = nn.DataParallel(model)

        return model, criterion

    @staticmethod
    def add_all_wrong_predictions(gts, predict_ret, output_fn: str, global_step: int = 0):
        """
        :param gts: size(n,) ndarray GroundTruth Label. Not index.
        :param predict_ret: list size(n). row: filename, predict_label, predict_probability
        :param output_fn:str:
        :param global_step: tensorboard global step.
        """
        if not output_fn:
            return

        if len(gts) != len(predict_ret):
            raise ValueError(f"GroundTruth len: [{len(gts)}] and predict len: [{len(predict_ret)}] mismatch.")

        all_details = []
        for i, gt_label in enumerate(gts):
            if gt_label != predict_ret[i][1]:
                all_details.append((predict_ret[i][0], gt_label, predict_ret[i][1], predict_ret[i][2]))

        all_details = sorted(all_details, key=lambda x: x[-1], reverse=True)

        for idx, row in enumerate(all_details):
            val = [str(x) for x in row]
            val.insert(0, str(idx + 1))
            all_details[idx] = val

        thead = ["Id", "filename", "GroundTruth", "Actual Pred", "Pred Prob"]

        with open(output_fn, "w+") as f:
            f.write("EP: " + str(global_step) + "\n")
            f.write("\t".join(thead) + "\n")
            for x in all_details:
                f.write("\t".join(x) + "\n")
