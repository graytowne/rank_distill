import argparse
from time import time

import torch.optim as optim

from caser import Caser
from train_caser import Recommender
from evaluation import evaluate_ranking
from interactions import Interactions
from losses import weighted_sigmoid_log_loss
from utils import *

import numpy as np
import torch
import os


class DistilledRecommender(Recommender):
    """
    Contains attributes and methods that needed to train a sequential
    recommendation model with ranking distillation[1]. Models are trained
    by many tuples of (users, sequences, targets, negatives) and negatives
    are from negative sampling: for any known tuple of (user, sequence, targets),
    one or more items are randomly sampled to act as negatives.

    [1] Ranking Distillation: Learning Compact Ranking Models With High
        Performance for Recommender System, Jiaxi Tang and Ke Wang , KDD '18

    Parameters
    ----------

    n_iter: int,
        Number of iterations to run.
    batch_size: int,
        Minibatch size.
    l2: float,
        L2 loss penalty, also known as the 'lambda' of l2 regularization.
    neg_samples: int,
        Number of negative samples to generate for each targets.
    learning_rate: float,
        Initial learning rate.
    use_cuda: boolean,
        Run the model on a GPU or CPU.
    teacher_model_path: string,
        Path to teacher's model checkpoint.
    teacher_topk_path: string,
        Path to teacher's top-K ranking cache for each training instance.
    lamda: float
        Hyperparameter for tuning the sharpness of position importance weight.
    mu: float
        Hyperparameter for tuning the sharpness of ranking discrepancy weight.
    num_dynamic_samples: int
        Number of samples used for estimating student's rank.
    dynamic_start_epoch: int
        Number of iteration to start using hybrid of two different weights.
    K: int
        Length of teacher's exemplary ranking.
    teach_alpha: float:
        Weight for balancing ranking loss and distillation loss.
    student_model_args: args,
        Student model related arguments, like latent dimensions.
    teacher_model_args: args,
        Teacher model related arguments, like latent dimensions.
    """
    def __init__(self,
                 n_iter=None,
                 batch_size=None,
                 l2=None,
                 neg_samples=None,
                 learning_rate=None,
                 use_cuda=False,
                 teacher_model_path=None,
                 teacher_topk_path=None,
                 lamda=None,
                 mu=None,
                 num_dynamic_samples=None,
                 dynamic_start_epoch=None,
                 K=None,
                 teach_alpha=None,
                 student_model_args=None,
                 teacher_model_args=None):

        # data related
        self.L = None
        self.T = None

        # model related
        self._num_items = None
        self._num_users = None
        self._teacher_net = None  # teacher model
        self._student_net = None  # student model
        self._student_model_args = student_model_args
        self._teacher_model_args = teacher_model_args

        # learning related
        self._batch_size = batch_size
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._neg_samples = neg_samples
        self._device = torch.device("cuda" if use_cuda else "cpu")

        # ranking distillation related
        self._teach_alpha = teach_alpha
        self._lambda = lamda
        self._mu = mu
        self._num_dynamic_samples = num_dynamic_samples
        self._dynamic_start_epoch = dynamic_start_epoch
        self._K = K
        self._teacher_model_path = teacher_model_path
        self._teacher_topk_path = teacher_topk_path
        self._weight_renormalize = False

        # rank evaluation related
        self.test_sequence = None
        self._candidate = dict()

    @property
    def _teacher_initialized(self):
        return self._teacher_net is not None

    def _initialize_teacher(self, interactions):
        # initialize teacher model
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users

        self._teacher_net = Caser(self._num_users,
                                  self._num_items,
                                  self._teacher_model_args)
        # load teacher model
        if os.path.isfile(self._teacher_model_path):
            output_str = ("loading teacher model from %s" % self._teacher_model_path)
            print(output_str)

            checkpoint = torch.load(self._teacher_model_path)
            self._teacher_net.load_state_dict(checkpoint['state_dict'])
            output_str = "loaded model %s (epoch %d)" % (self._teacher_model_path, checkpoint['epoch_num'])
            print(output_str)
        else:
            output_str = "no model found at %s" % self._teacher_model_path
            print output_str

        # set teacher model to evaluation mode
        self._teacher_net.eval()

    @property
    def _student_initialized(self):
        return self._student_net is not None

    def _initialize_student(self, interactions):
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users

        self.test_sequence = interactions.test_sequences

        self._student_net = Caser(self._num_users,
                                  self._num_items,
                                  self._student_model_args)

        self._optimizer = optim.Adam(self._student_net.parameters(),
                                     weight_decay=self._l2,
                                     lr=self._learning_rate)

    def fit(self, train, test, verbose=False):
        """
        The general training loop to fit the model

        Parameters
        ----------

        train: :class:`interactions.Interactions`
            training instances, also contains test sequences
        test: :class:`interactions.Interactions`
            only contains targets for test sequences
        verbose: bool, optional
            print the logs
        """

        # convert sequences, targets and users to numpy arrays
        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets
        users_np = train.sequences.user_ids.reshape(-1, 1)

        self.L, self.T = train.sequences.L, train.sequences.T

        n_train = sequences_np.shape[0]

        output_str = 'total training instances: %d' % n_train
        print(output_str)

        if not self._teacher_initialized:
            self._initialize_teacher(train)
        if not self._student_initialized:
            self._initialize_student(train)

        # here we compute teacher top-K ranking for each training instance in advance for faster training speed
        # while we have to compute the top-K ranking on the fly if it is too large to keep in memory
        if os.path.isfile(self._teacher_topk_path):
            print('found teacher topk file, loading..')
            teacher_ranking = np.load(self._teacher_topk_path)
        else:
            print('teacher topk file not found, generating.. ')
            teacher_ranking = self._get_teacher_topk(sequences_np, users_np, targets_np, k=self._K)

        # initialize static weight (position importance weight)
        weight_static = np.array(range(1, self._K + 1), dtype=np.float32)
        weight_static = np.exp(-weight_static / self._lambda)
        weight_static = weight_static / np.sum(weight_static)

        weight_static = torch.from_numpy(weight_static).to(self._device)
        weight_static = weight_static.unsqueeze(0)

        # initialize dynamic weight (ranking discrepancy weight)
        weight_dynamic = None

        # count number of parameters
        print("Number of params in teacher model: %d" % compute_model_size(self._teacher_net))
        print("Number of params in student model: %d" % compute_model_size(self._student_net))

        indices = np.arange(n_train)
        start_epoch = 1

        for epoch_num in range(start_epoch, self._n_iter + 1):

            t1 = time()
            # set teacher model to evaluation mode and move it to the corresponding devices
            self._teacher_net.eval()
            self._teacher_net = self._teacher_net.to(self._device)
            # set student model to training mode and move it to the corresponding devices
            self._student_net.train()
            self._student_net = self._student_net.to(self._device)

            (users_np, sequences_np, targets_np), shuffle_indices = shuffle(users_np,
                                                                            sequences_np,
                                                                            targets_np,
                                                                            indices=True)

            indices = indices[shuffle_indices]  # keep indices for retrieval teacher's top-K ranking from cache

            negatives_np = self._generate_negative_samples(users_np, train, n=self._neg_samples)

            dynamic_samples_np = self._generate_negative_samples(users_np, train, n=self._num_dynamic_samples)

            # convert numpy arrays to PyTorch tensors and move it to the corresponding devices
            users, sequences, targets, negatives = (torch.from_numpy(users_np).long(),
                                                    torch.from_numpy(sequences_np).long(),
                                                    torch.from_numpy(targets_np).long(),
                                                    torch.from_numpy(negatives_np).long())

            users, sequences, targets, negatives = (users.to(self._device),
                                                    sequences.to(self._device),
                                                    targets.to(self._device),
                                                    negatives.to(self._device))

            dynamic_samples = torch.from_numpy(dynamic_samples_np).long().to(self._device)

            epoch_loss = 0.0
            epoch_regular_loss = 0.0

            for (minibatch_num,
                 (batch_indices,
                  batch_users,
                  batch_sequences,
                  batch_targets,
                  batch_negatives,
                  batch_dynamics)) in enumerate(minibatch(indices,
                                                          users,
                                                          sequences,
                                                          targets,
                                                          negatives,
                                                          dynamic_samples,
                                                          batch_size=self._batch_size)):

                # retrieval teacher top-K ranking given indices
                batch_candidates = torch.from_numpy(teacher_ranking[batch_indices, :]).long().to(self._device)
                # concatenate all variables to get predictions in one run
                items_to_predict = torch.cat((batch_targets, batch_negatives,
                                              batch_candidates, batch_dynamics), 1)

                items_prediction = self._student_net(batch_sequences,
                                                     batch_users,
                                                     items_to_predict)

                (targets_prediction,
                 negatives_prediction,
                 candidates_prediction,
                 dynamics_prediction) = torch.split(items_prediction, [batch_targets.size(1),
                                                                       batch_negatives.size(1),
                                                                       batch_candidates.size(1),
                                                                       batch_dynamics.size(1)], dim=1)

                self._optimizer.zero_grad()

                if epoch_num > self._dynamic_start_epoch:
                    # compute dynamic weight
                    dynamic_weights = list()
                    for col in range(self._K):
                        col_prediction = candidates_prediction[:, col].unsqueeze(1)

                        num_smaller_than = torch.sum(col_prediction < dynamics_prediction, dim=1).float()
                        relative_rank = num_smaller_than / self._num_dynamic_samples
                        predicted_rank = torch.floor((self._num_items - 1) * relative_rank)

                        dynamic_weight = torch.tanh(self._mu * (predicted_rank - col))
                        dynamic_weight = torch.clamp(dynamic_weight, min=0.0)

                        dynamic_weights.append(dynamic_weight)
                    weight_dynamic = torch.stack(dynamic_weights, 1)

                    # hybrid two weights
                    weight = weight_dynamic * weight_static
                    if self._weight_renormalize:
                        weight = F.normalize(weight, p=1, dim=1)
                else:
                    weight = weight_static

                # detach the weight to stop the gradient flow to the weight
                weight = weight.detach()

                loss, regular_loss = weighted_sigmoid_log_loss(targets_prediction,
                                                               negatives_prediction,
                                                               candidates_prediction,
                                                               weight, self._teach_alpha)

                epoch_loss += loss.item()
                epoch_regular_loss += regular_loss.item()

                loss.backward()

                # assert False
                self._optimizer.step()

            epoch_loss /= minibatch_num + 1
            epoch_regular_loss /= minibatch_num + 1

            t2 = time()

            if verbose and epoch_num % 10 == 0:
                precision, recall, ndcg, mean_aps = evaluate_ranking(self, test, train, k=[3, 5, 10])

                str_precs = "precisions=%.4f,%.4f,%.4f" % tuple([np.mean(a) for a in precision])
                str_recalls = "recalls=%.4f,%.4f,%.4f" % tuple([np.mean(a) for a in recall])
                str_ndcgs = "ndcgs=%.4f,%.4f,%.4f" % tuple([np.mean(a) for a in ndcg])

                output_str = "Epoch %d [%.1f s]\tloss=%.4f, regular_loss=%.4f, " \
                             "map=%.4f, %s, %s, %s[%.1f s]" % (epoch_num, t2 - t1,
                                                               epoch_loss, epoch_regular_loss,
                                                               mean_aps, str_precs, str_recalls, str_ndcgs,
                                                               time() - t2)
                print(output_str)
            else:
                output_str = "Epoch %d [%.1f s]\tloss=%.4f, regular_loss=%.4f[%.1f s]" % (epoch_num, t2 - t1,
                                                                                          epoch_loss,
                                                                                          epoch_regular_loss,
                                                                                          time() - t2)
                print(output_str)

    def _get_teacher_topk(self, sequences, users, targets, k):
        """
        Pre-compute and cache teacher's top-K ranking for each training instance.
        By doing this we can make training with distillation much faster.

        Parameters
        ----------

        sequences: array of np.int64
            sequencces of items
        users: array of np.int64
            users associated with each sequence
        targets: array of np.int64
            target item that user interact with given the sequence
        k: int
            length of teacher's exemplary ranking
        """
        with_targets = False

        n_train = sequences.shape[0]
        indices = np.arange(n_train)

        users, sequences = torch.from_numpy(users).long(), torch.from_numpy(sequences).long()

        # teacher topk results
        teacher_topk = np.zeros((n_train, k), dtype=np.int64)

        for (batch_indices,
             batch_users,
             batch_sequences,
             batch_targets) in minibatch(indices,
                                         users,
                                         sequences,
                                         targets,
                                         batch_size=16):

            cur_batch_size = batch_users.shape[0]
            all_items = torch.arange(start=0, end=self._num_items).repeat(cur_batch_size, 1).long()

            teacher_prediction = self._teacher_net(batch_sequences,
                                                   batch_users,
                                                   all_items).detach()

            _, tops = teacher_prediction.topk(k * 2, dim=1)  # return the topk by column
            tops = tops.cpu().numpy()

            new_tops = np.concatenate((batch_targets, tops), axis=1)
            topks = np.zeros((cur_batch_size, k), dtype=np.int64)

            for i, row in enumerate(new_tops):
                _, idx = np.unique(row, return_index=True)
                # whether teacher's top-k ranking consider target items
                if with_targets:
                    topk = row[np.sort(idx)][:k]
                else:
                    topk = row[np.sort(idx)][self.T:k + self.T]
                topks[i, :] = topk
            teacher_topk[batch_indices, :] = topks
        np.save('gowalla-teacher-dim=%d-top=%d.npy' % (self._teacher_model_args.d, k), teacher_topk)
        return teacher_topk

    def predict(self, user_id, item_ids=None, model=None):
        return super(DistilledRecommender, self).predict(user_id, item_ids,
                                                         model=self._student_net)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--train_root', type=str, default='datasets/gowalla/test/train.txt')
    parser.add_argument('--test_root', type=str, default='datasets/gowalla/test/test.txt')
    parser.add_argument('--L', type=int, default=5)
    # train arguments
    parser.add_argument('--n_iter', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    # distillation arguments
    # dimensionality of teacher model (specifically for embedding)
    parser.add_argument('--teacher_model_dim', type=int, default=100)
    # path to teacher's model checkpoint
    parser.add_argument('--teacher_model_path', type=str, default='checkpoints/gowalla-caser-dim=100.pth.tar')
    # path to teacher's top-K ranking cache for each training instance
    parser.add_argument('--teacher_topk_path', type=str, default='')
    # here alpha=1.0 stands for equal weight for ranking loss and distillation loss
    parser.add_argument('--teach_alpha', type=float, default=1.0)
    # length of teacher's exemplary ranking
    parser.add_argument('--K', type=int, default=10)
    # hyperparameter for tuning the sharpness of position importance weight in Eq.(8)
    parser.add_argument('--lamda', type=float, default=1)
    # hyperparameter for tuning the sharpness of ranking discrepancy weight in Eq.(9)
    parser.add_argument('--mu', type=float, default=0.1)
    # number of samples used for estimating student's rank in Eq.(9)
    parser.add_argument('--num_dynamic_samples', type=int, default=100)
    # number of iteration to start using hybrid of two different weights
    parser.add_argument('--dynamic_start_epoch', type=int, default=10)

    config = parser.parse_args()

    # model dependent arguments
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--d', type=int, default=50)

    # Caser args
    model_parser.add_argument('--nv', type=int, default=2)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='iden')
    model_parser.add_argument('--ac_fc', type=str, default='sigm')

    teacher_model_config = model_parser.parse_args()
    teacher_model_config.L = config.L
    teacher_model_config.d = config.teacher_model_dim

    student_model_config = model_parser.parse_args()
    student_model_config.L = config.L

    # set seed
    set_seed(config.seed,
             cuda=config.use_cuda)

    train = Interactions(config.train_root)
    # transform triplets to sequence representation
    train.to_sequence(config.L)

    test = Interactions(config.test_root,
                        user_map=train.user_map,
                        item_map=train.item_map)

    print(config)
    print(student_model_config)
    # fit model
    model = DistilledRecommender(n_iter=config.n_iter,
                                 batch_size=config.batch_size,
                                 learning_rate=config.learning_rate,
                                 l2=config.l2,
                                 use_cuda=config.use_cuda,
                                 neg_samples=config.neg_samples,
                                 teacher_model_path=config.teacher_model_path,
                                 teacher_topk_path=config.teacher_topk_path,
                                 teacher_model_args=teacher_model_config,
                                 student_model_args=student_model_config,
                                 lamda=config.lamda,
                                 mu=config.mu,
                                 num_dynamic_samples=config.num_dynamic_samples,
                                 dynamic_start_epoch=config.dynamic_start_epoch,
                                 K=config.K,
                                 teach_alpha=config.teach_alpha)

    model.fit(train, test, verbose=True)
