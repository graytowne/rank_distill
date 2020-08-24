import argparse
from time import time

import torch.optim as optim

from caser import Caser
from evaluation import evaluate_ranking
from interactions import Interactions
from losses import sigmoid_log_loss
from utils import *


def _save_checkpoint(state, filename):
    print("Saving checkpoint to %s." % filename)
    torch.save(state, filename)


class Recommender(object):
    """
    Contains attributes and methods that needed to train a sequential
    recommendation model. Models are trained by many tuples of
    (users, sequences, targets, negatives) and negatives are from negative
    sampling: for any known tuple of (user, sequence, targets), one or more
    items are randomly sampled to act as negatives.


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
        If targets=3 and neg_samples=3, then it will sample 9 negatives.
    learning_rate: float,
        Initial learning rate.
    use_cuda: boolean,
        Run the model on a GPU or CPU.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self,
                 n_iter=None,
                 batch_size=None,
                 l2=None,
                 neg_samples=None,
                 learning_rate=None,
                 use_cuda=False,
                 checkpoint=None,
                 model_args=None):

        # model related
        self._num_items = None
        self._num_users = None
        self._net = None
        self.model_args = model_args

        # learning related
        self._batch_size = batch_size
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._neg_samples = neg_samples
        self._device = torch.device("cuda" if use_cuda else "cpu")
        self.checkpoint = checkpoint

        # rank evaluation related
        self.test_sequence = None
        self._candidate = dict()

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users

        self.test_sequence = interactions.test_sequences

        self._net = Caser(self._num_users,
                          self._num_items,
                          self.model_args)

        self._optimizer = optim.Adam(
            self._net.parameters(),
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

        L, T = train.sequences.L, train.sequences.T

        n_train = sequences_np.shape[0]

        print('total training instances: %d' % n_train)

        if not self._initialized:
            self._initialize(train)

        start_epoch = 1
        if self.checkpoint:
            print("loading checkpoint from %s" % self.checkpoint)
            checkpoint = torch.load(self.checkpoint)
            start_epoch = checkpoint['epoch_num']
            self._net.load_state_dict(checkpoint['state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer'])
            print("loaded checkpoint %s (epoch %d)" % (self.checkpoint, start_epoch))

        # compute number of parameters
        print("Number of params: %d" % compute_model_size(self._net))

        for epoch_num in range(start_epoch, self._n_iter + 1):

            t1 = time()

            # set model to training model and move it to the corresponding devices
            self._net.train()
            self._net = self._net.to(self._device)

            users_np, sequences_np, targets_np = shuffle(users_np,
                                                         sequences_np,
                                                         targets_np)

            negatives_np = self._generate_negative_samples(users_np, train, n=self._neg_samples)

            # convert numpy arrays to PyTorch tensors and move it to the corresponding devices
            users, sequences, targets, negatives = (torch.from_numpy(users_np).long(),
                                                    torch.from_numpy(sequences_np).long(),
                                                    torch.from_numpy(targets_np).long(),
                                                    torch.from_numpy(negatives_np).long())

            users, sequences, targets, negatives = (users.to(self._device),
                                                    sequences.to(self._device),
                                                    targets.to(self._device),
                                                    negatives.to(self._device))

            epoch_loss = 0.0

            for (minibatch_num,
                 (batch_users,
                  batch_sequences,
                  batch_targets,
                  batch_negatives)) in enumerate(minibatch(users,
                                                           sequences,
                                                           targets,
                                                           negatives,
                                                           batch_size=self._batch_size)):
                # concatenate all variables to get predictions in one run
                items_to_predict = torch.cat((batch_targets, batch_negatives), 1)
                items_prediction = self._net(batch_sequences,
                                             batch_users,
                                             items_to_predict)

                (targets_prediction,
                 negatives_prediction) = torch.split(items_prediction, [batch_targets.size(1),
                                                                        batch_negatives.size(1)], dim=1)

                self._optimizer.zero_grad()
                # compute the binary cross-entropy loss
                loss = sigmoid_log_loss(targets_prediction, negatives_prediction)

                epoch_loss += loss.item()

                loss.backward()
                self._optimizer.step()

            epoch_loss /= minibatch_num + 1

            t2 = time()
            if verbose and epoch_num % 10 == 0:
                precision, recall, ndcg, mean_aps = evaluate_ranking(self, test, train, k=[3, 5, 10])
                str_precs = "precisions=%.4f,%.4f,%.4f" % tuple([np.mean(a) for a in precision])
                str_recalls = "recalls=%.4f,%.4f,%.4f" % tuple([np.mean(a) for a in recall])
                str_ndcgs = "ndcgs=%.4f,%.4f,%.4f" % tuple([np.mean(a) for a in ndcg])

                output_str = "Epoch %d [%.1f s]\tloss=%.4f, " \
                             "map=%.4f, %s, %s, %s[%.1f s]" % (epoch_num, t2 - t1,
                                                               epoch_loss,
                                                               mean_aps, str_precs, str_recalls, str_ndcgs,
                                                               time() - t2)
                print(output_str)
            else:
                output_str = "Epoch %d [%.1f s]\tloss=%.4f [%.1f s]" % (epoch_num,
                                                                        t2 - t1,
                                                                        epoch_loss,
                                                                        time() - t2)
                print(output_str)

        _save_checkpoint({
            'epoch_num': epoch_num,
            'state_dict': self._net.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }, 'checkpoints/gowalla-caser-dim=%d.pth.tar' % self.model_args.d)

    def _generate_negative_samples(self, users, interactions, n):
        """
        Sample negative from a candidate set of each user. The
        candidate set of each user is defined by:
        {All Items} \ {Items Rated by User}

        Parameters
        ----------

        users: array of np.int64
            sequence users
        interactions: :class:`interactions.Interactions`
            training instances, used for generate candidates
        n: int
            total number of negatives to sample for each sequence
        """

        users_ = users.squeeze()
        negative_samples = np.zeros((users_.shape[0], n), np.int64)
        if not self._candidate:
            all_items = np.arange(interactions.num_items - 1) + 1  # 0 for padding
            train = interactions.tocsr()
            for user, row in enumerate(train):
                self._candidate[user] = list(set(all_items) - set(row.indices))

        for i, u in enumerate(users_):
            for j in range(n):
                x = self._candidate[u]
                negative_samples[i, j] = x[
                    np.random.randint(len(x))]

        return negative_samples

    def predict(self, user_id, item_ids=None, model=None):
        """
        Make predictions for evaluation: given a user id, it will
        first retrieve the test sequence associated with that user
        and compute the recommendation scores for items.

        Parameters
        ----------

        user_id: int
           users id for which prediction scores needed.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.
        """

        if self.test_sequence is None:
            raise ValueError('Missing test sequences, cannot make predictions')
        if model is None:
            model = self._net

        # set model to evaluation model
        model.eval()
        with torch.no_grad():
            sequences_np = self.test_sequence.sequences[user_id, :]
            sequences_np = np.atleast_2d(sequences_np)

            if item_ids is None:
                item_ids = np.arange(self._num_items).reshape(-1, 1)

            sequences = torch.from_numpy(sequences_np.astype(np.int64).reshape(1, -1))
            item_ids = torch.from_numpy(item_ids.astype(np.int64))
            user_id = torch.from_numpy(np.array([[user_id]]).astype(np.int64))

            user, sequences, items = (user_id.to(self._device),
                                      sequences.to(self._device),
                                      item_ids.to(self._device))

            out = model(sequences,
                        user,
                        items,
                        for_pred=True)

            return cpu(out.data).numpy().flatten()


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
    parser.add_argument('--checkpoint', type=str, default='')

    config = parser.parse_args()

    # model dependent arguments
    # see https://github.com/graytowne/caser_pytorch for more details about these arguments.
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--d', type=int, default=100)
    model_parser.add_argument('--nv', type=int, default=2)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='iden')
    model_parser.add_argument('--ac_fc', type=str, default='sigm')

    model_config = model_parser.parse_args()
    model_config.L = config.L

    # set seed
    set_seed(config.seed,
             cuda=config.use_cuda)

    # load dataset
    train = Interactions(config.train_root)
    # transform triplets to sequence representation
    train.to_sequence(config.L)

    test = Interactions(config.test_root,
                        user_map=train.user_map,
                        item_map=train.item_map)

    print(config)
    print(model_config)
    # fit model
    model = Recommender(n_iter=config.n_iter,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate,
                        l2=config.l2,
                        neg_samples=config.neg_samples,
                        use_cuda=config.use_cuda,
                        checkpoint=config.checkpoint,
                        model_args=model_config)

    model.fit(train, test, verbose=True)
