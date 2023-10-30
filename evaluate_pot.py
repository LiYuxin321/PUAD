# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import time
# from logger import Logger
from spot import SPOT, bidSPOT
from pprint import pformat, pprint


class Evaluator():
    def __init__(self, anomaly_score_label_file, bf_search_min=-50, bf_search_max=5,
                 bf_search_step_size=0.2, level=0.0050, log_path='', log_file=''):
        self.anomaly_score_label_file = anomaly_score_label_file
        self.bf_search_min = bf_search_min
        self.bf_search_max = bf_search_max
        self.bf_search_step_size = bf_search_step_size
        self.level = level

        self.log_path = log_path
        self.log_file = log_file

        # self.logger = Logger(self.log_path, self.log_file)

    def get_label_0_1(self, timestamp_anomalyscore_label):
        label = []
        for idx in range(len(timestamp_anomalyscore_label[2])):
            if timestamp_anomalyscore_label[2][idx] == "Anomaly":
                label.append(1)
            else:
                label.append(0)
        return np.array(label)

    def perform_evaluating(self):

        timestamp_anomalyscore_label1 = np.loadtxt(self.anomaly_score_label_file, delimiter=',', dtype=bytes,
                                                   unpack=False).astype(str)
        timestamp_anomalyscore_label2 = timestamp_anomalyscore_label1.tolist()
        timestamp_anomalyscore_label2.sort()
        timestamp_anomalyscore_label3 = [[], [], []]
        for i in range(len(timestamp_anomalyscore_label2)):
            timestamp_anomalyscore_label3[0].append(timestamp_anomalyscore_label2[i][0])
            timestamp_anomalyscore_label3[1].append(timestamp_anomalyscore_label2[i][1])
            timestamp_anomalyscore_label3[2].append(timestamp_anomalyscore_label2[i][2])
        timestamp_anomalyscore_label = np.array(timestamp_anomalyscore_label3)

        label_0_1 = self.get_label_0_1(timestamp_anomalyscore_label)

        '''
        Get the f1 score via POT
        '''
        t, th = bf_search(timestamp_anomalyscore_label[1].astype(np.float),
                          label_0_1,
                          start=self.bf_search_min,
                          end=self.bf_search_max,
                          step_num=int(abs(self.bf_search_max - self.bf_search_min) /
                                       self.bf_search_step_size),
                          display_freq=5)
        best_valid_metrics = {}
        # output the results
        best_valid_metrics.update({
            'best-f1': t[0],
            'precision': t[1],
            'recall': t[2],
            'TP': t[3],
            'TN': t[4],
            'FP': t[5],
            'FN': t[6],
            'latency': t[-1],
            'threshold': th
        })

        pot_result = pot_eval(timestamp_anomalyscore_label[1].astype(np.float),
                              timestamp_anomalyscore_label[1].astype(np.float), label_0_1, level=self.level)
        best_valid_metrics.update(pot_result)
        pprint(best_valid_metrics)
        self.logger.log_evaluator_pot('The current level: {}'.format(self.level))
        self.logger.log_evaluator_pot(best_valid_metrics)


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).


    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        # if verbose and i % display_freq == 0:
        #     print("cur thr: ", threshold, target, m, m_t)
    # print(m, m_t)
    return m, m_t


def pot_eval(init_score, score, label, q=1e-3, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            For `OmniAnomaly`, it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            For `OmniAnomaly`, it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t

    Returns:
        dict: pot result dict
    """
    s = SPOT(q)  # SPOT object bidSPOT SPOT
    s.fit(init_score, score)  # data import
    s.initialize(level=level, min_extrema=True)  # initialization step
    ret = s.run(dynamic=False)  # run

    # s = bidSPOT(q)  # SPOT object bidSPOT SPOT
    # s.fit(init_score, score)  # data import
    # s.initialize(level=level)  # initialization step
    # ret = s.run()  # run

    # print(len(ret['alarms']))
    # print(len(ret['thresholds']))
    pot_th = -np.mean(ret['thresholds'])  # upper_thresholds lower_thresholds
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    p_t = calc_point2point(pred, label)
    # print('POT result: ', p_t, pot_th, p_latency)
    return {
        'pot-f1': p_t[0],
        'pot-precision': p_t[1],
        'pot-recall': p_t[2],
        'pot-TP': p_t[3],
        'pot-TN': p_t[4],
        'pot-FP': p_t[5],
        'pot-FN': p_t[6],
        'pot-threshold': pot_th,
        'pot-latency': p_latency,
        'pred': pred
    }


def main():
    parser = argparse.ArgumentParser()
    # GPU option
    parser.add_argument('--gpu_id', type=int, default=0)
    # Dataset options
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--T', type=int, default=20)
    parser.add_argument('--win_size', type=int, default=1)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--n', type=int, default=36)

    # Model options
    parser.add_argument('--categorical_dims', type=int, default=5)
    parser.add_argument('--z_dims', type=int, default=10)
    parser.add_argument('--conv_dims', type=int, default=20)
    parser.add_argument('--hidden_dims', type=int, default=20)
    parser.add_argument('--enc_dec', type=str, default='CNN')

    # Training options
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--temperature', type=float, default=5.0)
    parser.add_argument('--min_temperature', type=float, default=0.1)
    parser.add_argument('--anneal_rate', type=float, default=0.1)
    parser.add_argument("--hard_gumbel", action='store_true', help='hard gumbel or not')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--start_epoch', type=int, default=20)
    parser.add_argument('--checkpoints_path', type=str, default='')
    parser.add_argument('--checkpoints_file', type=str, default='')
    parser.add_argument('--checkpoints_interval', type=int, default=1)
    parser.add_argument('--log_path', type=str, default='log_evaluator_pot')
    parser.add_argument('--log_file', type=str, default='')

    parser.add_argument('--llh_path', type=str, default='log_tester')
    parser.add_argument('--llh_file', type=str, default='')

    parser.add_argument('--level', type=float, default=0.0030)
    parser.add_argument('--bf_search_min', type=float, default=-50.0)
    parser.add_argument('--bf_search_max', type=float, default=10.0)
    parser.add_argument('--bf_search_step_size', type=float, default=0.2)

    args = parser.parse_args()

    if args.enc_dec == 'CNN':
        if args.llh_file == '':
            args.llh_file = 'catdim{}_zdim{}_cdim{}_hdim{}_winsize{}_T{}_l{}_epochs{}_loss.txt'.format(
                args.categorical_dims,
                args.z_dims,
                args.conv_dims,
                args.hidden_dims,
                args.win_size,
                args.T,
                args.l,
                args.start_epoch)

        if args.log_file == '':
            args.log_file = 'catdim{}_zdim{}_cdim{}_hdim{}_winsize{}_T{}_l{}_epochs{}_eval_records'.format(
                args.categorical_dims,
                args.z_dims,
                args.conv_dims,
                args.hidden_dims,
                args.win_size,
                args.T,
                args.l,
                args.start_epoch)

    else:
        raise ValueError('Unknown encoder and decoder: {}'.format(args.enc_dec))

    if not os.path.exists(os.path.join(args.llh_path, args.llh_file)):
        raise ValueError('Unknown anomaly score label file: {}/{}'.format(args.llh_path, args.llh_file))

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    anomaly_score_label_file = os.path.join(args.llh_path, args.llh_file)
    training_dataset_anomaly_score_label_file = os.path.join(args.llh_path, args.llh_file)
    evaluator = Evaluator(anomaly_score_label_file,
                          level=args.level,
                          log_path=args.log_path, log_file=args.log_file)

    evaluator.perform_evaluating()


if __name__ == '__main__':
    main()
