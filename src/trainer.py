# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
import scipy
import scipy.linalg
import numpy
import torch
from torch.autograd import Variable, Function
from torch.nn import functional as F
from torch.nn import Parameter

from .utils import get_optimizer, export_embeddings
from .utils import clip_parameters
from .dico_builder import build_dictionary
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary


logger = getLogger()


class Trainer(object):

    def __init__(self, src_emb, tgt_emb, mapping, discriminator, generator, params):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        self.mapping = mapping
        self.discriminator = discriminator
        self.generator = generator
        self.params = params

        # optimizers
        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            self.map_optimizer = optim_fn(mapping.parameters(), **optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_params)
        if hasattr(params, 'gen_optimizer'):
            optim_fn, optim_params = get_optimizer(params.gen_optimizer)
            self.gen_optimizer = optim_fn(generator.parameters(), **optim_params)
        else:
            assert discriminator is None

        # best validation score
        self.best_valid_metric = -1e12

        self.decrease_lr = False

        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1
        self.one = Variable(self.one.cuda() if self.params.cuda else self.one)
        self.mone = Variable(self.mone.cuda() if self.params.cuda else self.mone)

    def get_dis_xy(self, volatile):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        assert mf <= min(len(self.src_dico), len(self.tgt_dico))
        src_ids = torch.LongTensor(bs).random_(len(self.src_dico) if mf == 0 else mf)
        tgt_ids = torch.LongTensor(bs).random_(len(self.tgt_dico) if mf == 0 else mf)
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        src_emb = self.src_emb(Variable(src_ids, volatile=True))
        tgt_emb = self.tgt_emb(Variable(tgt_ids, volatile=True))
        src_emb = self.mapping(Variable(src_emb.data, volatile=volatile))
        tgt_emb = Variable(tgt_emb.data, volatile=volatile)

        # input / target
        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.FloatTensor(2 * bs).zero_()
        y[:bs] = 1 - self.params.dis_smooth
        y[bs:] = self.params.dis_smooth
        y = Variable(y.cuda() if self.params.cuda else y)

        return x, y

    def get_wgan_dis_xy(self, volatile):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        assert mf <= min(len(self.src_dico), len(self.tgt_dico))
        src_ids = torch.LongTensor(bs).random_(len(self.src_dico) if mf == 0 else mf)
        tgt_ids = torch.LongTensor(bs).random_(len(self.tgt_dico) if mf == 0 else mf)
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        src_emb = self.src_emb(Variable(src_ids, volatile=True))
        tgt_emb = self.tgt_emb(Variable(tgt_ids, volatile=True))
        src_emb = self.mapping(Variable(src_emb.data, volatile=volatile)) # get Wx + b, training W.
        tgt_emb = Variable(tgt_emb.data, volatile=volatile)

        # input / target
        x = src_emb # * noise
        y = tgt_emb

        return x, y, src_ids, tgt_ids

    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()

        # loss
        x, y = self.get_dis_xy(volatile=True)
        preds = self.discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y)
        stats['DIS_COSTS'].append(loss.data[0])

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (discriminator)")
            exit()

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        clip_parameters(self.discriminator, self.params.dis_clip_weights)

    def wgan_dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()
        self.generator.eval()

        # loss
        x, y, _, _ = self.get_wgan_dis_xy(volatile=True)
        errD_real = self.discriminator(Variable(y.data))
        errD_real.backward(self.one)

        fake = self.generator(x.data)
        errD_fake = self.discriminator(fake)
        errD_fake.backward(self.mone)

        #loss = F.binary_cross_entropy(fake, real)
        errD = errD_real - errD_fake
        stats['DIS_COSTS'].append(errD.data.cpu().numpy()[0])

        # optim
        self.dis_optimizer.zero_grad()
        #loss.backward()
        self.dis_optimizer.step()
        clip_parameters(self.discriminator, self.params.dis_clip_weights)

    def mapping_step(self, stats):
        """
        Fooling discriminator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        self.discriminator.eval()

        # loss
        x, y = self.get_dis_xy(volatile=False)
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
        loss = self.params.dis_lambda * loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (fool discriminator)")
            exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()
        self.orthogonalize()

        return 2 * self.params.batch_size

    def wgan_generator_step(self, stats):
        """
        Fooling generator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        self.discriminator.eval()
        self.generator.train()

        # loss
        x, y, _, _ = self.get_wgan_dis_xy(volatile=False)
        fake = self.generator(x.data)
        errG = self.discriminator(fake)
        errG.backward(self.one)

        # optim
        self.gen_optimizer.zero_grad()
        self.gen_optimizer.step()
        self.orthogonalize()

        return 2 * self.params.batch_size

    def load_training_dico(self, dico_train):
        """
        Load training dictionary.
        """
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

        # identical character strings
        if dico_train == "identical_char":
            self.dico = load_identical_char_dico(word2id1, word2id2)
        # use one of the provided dictionary
        elif dico_train == "default":
            filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
            self.dico = load_dictionary(
                os.path.join(DIC_EVAL_PATH, filename),
                word2id1, word2id2
            )
        # dictionary provided by the user
        else:
            self.dico = load_dictionary(dico_train, word2id1, word2id2)

        # cuda
        if self.params.cuda:
            self.dico = self.dico.cuda()

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico = build_dictionary(src_emb, tgt_emb, self.params)

    def procrustes(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]
        W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

        """
        def procrustes_wgan(self, T):
            A = self.src_emb.weight.data[self.dico[:, 0]]
            B = self.tgt_emb.weight.data[self.dico[:, 1]]
            W = self.mapping.weight.data
            M = self.T.mm(B.transpose(0, 1).mm(A)).cpu().numpy()
            U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
            W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))
        """

    def sinkhorn(self):
        x, y, src_ids, tgt_ids = self.get_wgan_dis_xy(volatile=True)

        fake = self.generator(x.data)
        fake = fake.sum(1).expand_as(torch.ones(fake.size(0), fake.size(0)))
        target = y.sum(1).expand_as(torch.ones(y.size(0), y.size(0)))
        dist = ((fake ** 2) + (target ** 2) - 2 * fake * target).sqrt()

        # Approximate weights given emebdding index, since embeddings are sorted by frequency
        # TODO: get weights from FastText bin
        src_weight_sum = (len(self.src_emb.weight.data) * (len(self.src_emb.weight.data) + 1)) / 2
        tgt_weight_sum = (len(self.tgt_emb.weight.data) * (len(self.tgt_emb.weight.data) + 1)) / 2

        src_weights = src_ids.float() / float(src_weight_sum)
        tgt_weights = tgt_ids.float() / float(tgt_weight_sum)

        if self.params.cuda:
            src_weights = src_weights.cuda()
            tgt_weights = tgt_weights.cuda()

        self.sinkhorn = SinkHornAlgorithm(dist)

        dist_loss = self.sinkhorn(Variable(src_weights, volatile=True), Variable(tgt_weights, volatile=True))
        dist_loss.backward()
        self.sinkhorn.zero_grad()
        self.sinkhorn.step()

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        if self.params.map_beta > 0:
            W = self.mapping.weight.data
            beta = self.params.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.map_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True

    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            # save the mapping
            W = self.mapping.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_mapping.t7')
            logger.info('* Saving the mapping to %s ...' % path)
            torch.save(W, path)

    def reload_best(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params.exp_path, 'best_mapping.t7')
        logger.info('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        W = self.mapping.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))

    def export(self):
        """
        Export embeddings to a text file.
        """
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        export_embeddings(src_emb.cpu().numpy(), tgt_emb.cpu().numpy(), self.params)

class SinkHornAlgorithm(Function):
    """
    This code was built on top of Thomas Viehmann's Batch Sinkhorn Iteration Wasserstein Distance, https://github.com/t-vi/pytorch-tvmisc/wasserstein-distance/Pytorch_Wasserstein.ipynb
    """
    def __init__(self, dist, lam=1e-3, sinkhorn_iter=50):
        super(SinkHornAlgorithm, self).__init__()

        self.dist = dist
        self.lam = lam
        self.sinkhorn_iter = sinkhorn_iter
        self.na = dist.size(0)
        self.nb = dist.size(1)
        self.K = torch.exp(-self.dist/self.lam)
        self.KM = self.dist * self.K
        self.stored_grad = None
        
    def forward(self, src_weights, tgt_weight):
        """
        src_weights: source embedding mass points/frequence
        tgt_weight: target embedding mass points/frequence

        TODO: this is probably not optimal. Re-implement?
        """
        #import pdb; pdb.set_trace()
        assert src_weights.size(0) == self.na
        assert tgt_weight.size(0) == self.nb

        nbatch = src_weights.size(0)
        
        u = self.dist.new(nbatch, self.na).fill_(1.0/self.na)
        
        for i in range(self.sinkhorn_iter):
            v = tgt_weight / (torch.mm(u, self.K.t()))
            u = src_weights / (torch.mm(v, self.K))
            if (u!=u).sum()>0 or (v!=v).sum()>0 or u.max()>1e9 or v.max()>1e9:
                raise Exception(str(('Warning: numerical errrors', i+1, "u", (u!=u).sum(), u.max(), "v",(v!=v).sum(), v.max())))

        loss = (u * torch.mm(v, self.KM.t())).mean(0).sum()
        grad = self.lam * u.log()/nbatch 
        grad = grad-torch.mean(grad, dim=1).expand_as(grad)
        grad = grad-torch.mean(grad, dim=1).expand_as(grad)
        self.stored_grad = grad

        dist_loss = self.dist.new((loss,))
        return dist_loss
    
    def backward(self, grad_output):
        return self.stored_grad * grad_output[0], None
