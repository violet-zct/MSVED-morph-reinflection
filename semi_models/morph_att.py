__author__ ="chuntingzhou"
import logging

import emolga.basic.optimizers as optimizers
from emolga.layers.embeddings import *
from emolga.models.encdec import Encoder, DecoderAttCxt
from emolga.models.core import Model
from semi_models.layers import *
from semi_models.utils import *

logger = logging.getLogger(__name__)
RNN = GRU  # change it here for other RNN semi_models.


class SSL_VAE(Model):
    def __init__(self, srng, config):
        super(SSL_VAE, self).__init__()
        self.rng = srng
        self.alpha = config['alpha']
        self.addition_sup = config['add_sup']
        self.y_dim = config['y_dense_p_dim']
        self.z_dim = config['q_z_dim']

        # source and target word share the same encoder and decoder
        logger.info("build the SSL variational encoder-decoder")
        self.encoder = Encoder(config, self.rng, prefix='enc')

        if config['shared_embed']:
            self.decoder = DecoderAttCxt(config, self.rng, prefix='dec', mode='RNN', embed=self.encoder.Embed, init_h=True)
        else:
            self.decoder = DecoderAttCxt(config, self.rng, prefix='dec', mode='RNN', init_h=True)

        self.optimizer = optimizers.get(config['optimizer'])
        self.cross_only = config['cross_only']
        self.pure_sup = config['pure_sup']
        self.only_ul = config['only_ul']
        self.class_num = config['class_num']
        self.label_list = config['label_list']  # number of labels for each class

        self.dt_uns = config['dt_uns']

        self.activation_dense = config['activation_dense']  # 'tanh'
        self.init_w = config['init_dense']  # 'normal' or 'glorot_normal'
        self.both_gaussian = config['both_gaussian']

        self.enc_hidden_dim = config['enc_hidden_dim']
        self.dec_hidden_dim = config['dec_hidden_dim']
        self.dec_contxt_dim = config['dec_contxt_dim']

        self.has_ly_src = config['has_ly_src']
        self.bidirectional = config['bidirectional']
        self.semi_supervise = not (config['only_sup'] or config['only_ul'])

        self.z_mu = Dense(self.enc_hidden_dim if not self.bidirectional else 2 * self.enc_hidden_dim,
                                self.z_dim,
                                init=self.init_w,
                                activation='linear',
                                name='prior_z_mu')
        self.z_logvar = Dense(self.enc_hidden_dim if not self.bidirectional else 2 * self.enc_hidden_dim,
                                    self.z_dim,
                                    init=self.init_w,
                                    activation='linear',
                                    name='prior_z_logvar')

        self.y_dense_p = DenseN(self.label_list,
                                self.y_dim,
                                init=self.init_w,
                                activation=self.activation_dense,
                                name='dense_y_p')
        self.p_h_yz = Dense2(self.z_dim,
                             self.y_dim,
                             self.dec_hidden_dim,
                             init=self.init_w,
                             activation=self.activation_dense,
                             name='dense_h_yz')  # what activation here?
        # p(y)
        self.p_y = Log_py_prior(self.label_list)

        # q(y|x)
        self.q_y_x = q_y_x(self.enc_hidden_dim if not self.bidirectional else 2 * self.enc_hidden_dim,
                           self.label_list,
                           self.init_w,
                           self.activation_dense
                           )

        self._add(self.encoder)
        self._add(self.decoder)
        self._add(self.y_dense_p)
        self._add(self.z_mu)
        self._add(self.z_logvar)
        self._add(self.p_h_yz)
        self._add(self.p_y)
        self._add(self.q_y_x)

        if config['reload']:
            logging.info("Reload from %s", config['loadfrom'])
            self.load(config['loadfrom'])

    def _add(self, layer):
        if layer:
            self.layers.append(layer)
            self.params += layer.params

    def compile_gen(self):
        input = T.matrix('x', dtype='int32')  # padded input word for training
        target_label = [T.matrix('y', dtype='float32') for _ in range(0, self.class_num)]
        qx = self.encoder.build_encoder(input)
        z = self.z_mu(qx)
        dec_init_y, cxt, cxt_mask = self.y_dense_p(target_label)
        dec_init_state = self.p_h_yz(z, dec_init_y)  # (1, dec_hidden_dim)
        self.f_get_dec_inith_cxt = theano.function([input] + target_label, [dec_init_state, cxt, cxt_mask, z])
        self.decoder.build_sampler()

        if not self.pure_sup:
            _, logits = self.q_y_x(qx)
            sampled_y = []
            for i in range(0, self.class_num):
                sampled_y.append(gumbel_softmax(logits[i], 0.5, self.rng, True))
            self.f_get_gumbel_labels = theano.function([input], sampled_y)
            self.f_get_z = theano.function([input], z)

    def compile_train(self):
        input_sl = T.matrix('x_sl', dtype='int32')  # padded input word for training
        y_sl = [T.matrix('y_sl', dtype='float32') for _ in range(0, self.class_num)]
        input_tl = T.matrix('x_tl', dtype='int32')  # padded input word for training
        y_tl = [T.matrix('y_tl', dtype='float32') for _ in range(0, self.class_num)]
        input_u = T.matrix('x_u', dtype='int32')
        kl_anneal = T.scalar('kl_weight', dtype=theano.config.floatX)
        temp_anneal = T.scalar('temperature', dtype=theano.config.floatX)
        sl_anneal = T.scalar('sl_weight', dtype=theano.config.floatX)

        if self.pure_sup:
            l_obj, l_q_y_x_loss, kl_l = self.compile_train_l_one_direction(input_sl, input_tl, y_tl, kl_anneal)
            logging.info("Supervise Learning.")
            loss = l_obj
            updates = self.optimizer.get_updates(self.params, loss)
            logging.info("Compiling train function starts.............")
            self.train_ = theano.function(
                [input_sl] + [input_tl] + y_tl + [kl_anneal],
                [loss, l_q_y_x_loss, kl_l],
                updates=updates,
                name='train_fun',
                on_unused_input='ignore')
            return "Compilation of supervised function done."
        elif self.has_ly_src:
            l_obj, l_q_y_x_loss, l_inner, l_recon_loss, l_py, kl_l = self.compile_train_l(input_sl, input_tl, y_tl, temp_anneal, kl_anneal, y_sl)
        else:
            l_obj, l_q_y_x_loss, l_inner, l_recon_loss, l_py, kl_l = self.compile_train_l(input_sl, input_tl, y_tl, temp_anneal, kl_anneal)

        if not self.pure_sup:
            u_obj, u_inner, u_recon_loss, u_py, kl_u, att_probs = self.compile_train_u(input_u, kl_anneal, temp_anneal)

        if self.semi_supervise:
            logging.info("Semi-supervise Learning.")
            loss = sl_anneal * l_obj + self.dt_uns * u_obj
        elif self.only_ul:
            logging.info("Unsupervise Learning.")
            loss = u_obj
        else:
            logging.info("Weak/Supervise Learning.")
            loss = l_obj

        updates = self.optimizer.get_updates(self.params, loss)

        logging.info("Compiling train function starts.............")

        if self.has_ly_src:
            self.train_ = theano.function(
                [input_sl] + y_sl + [input_tl] + y_tl + [input_u] + [kl_anneal] + [temp_anneal] + [sl_anneal],
                [loss, l_obj, l_q_y_x_loss, u_obj, l_recon_loss, u_recon_loss, kl_l,
                 kl_u] + l_py + u_py,
                updates=updates,
                name='train_fun',
                on_unused_input='ignore')
        else:
            self.train_ = theano.function(
                [input_sl] + [input_tl] + y_tl + [input_u] + [kl_anneal] + [temp_anneal] + [sl_anneal],
                [loss, l_obj, l_q_y_x_loss, u_obj, l_recon_loss, u_recon_loss, kl_l,
                 kl_u] + l_py + u_py,
                updates=updates,
                name='train_fun',
                on_unused_input='ignore')
        logging.info("Compiling train function done.")
        self.test_pred_ = theano.function([input_tl] + y_tl + [kl_anneal], [l_q_y_x_loss] + l_py,
                                          on_unused_input='ignore')

    def compile_train_l_one_direction(self, input_sl, input_tl, y_tl, kl_anneal):
        batch_size = input_sl.shape[0]

        enc_src = self.encoder.build_encoder(input_sl)
        z_s_mu = self.z_mu(enc_src)
        z_s_logvar = self.z_logvar(enc_src)
        z_s = z_s_mu + T.exp(0.5 * z_s_logvar) * self.rng.normal(z_s_mu.shape, avg=0.0, std=1.0)

        dec_init_y_tgt, y_ctx_tgt, y_ctx_mask_tgt = self.y_dense_p(y_tl)
        dec_init_state_src_to_tgt = self.p_h_yz(z_s, dec_init_y_tgt)
        log_p_x_src_tgt, _, att_probs_src_tgt = self.decoder.build_decoder(input_tl, context=y_ctx_tgt,
                                                                           c_mask=y_ctx_mask_tgt, fixed_context=z_s,
                                                                           provide_init_h=dec_init_state_src_to_tgt)
        log_p_x_src_tgt = T.mean(-log_p_x_src_tgt.reshape((batch_size, 1)))
        log_p_y = T.mean(self.p_y(y_tl))
        kl_s = T.mean(-0.5 * T.sum(
            tensor.alloc(1.0, z_s_mu.shape[0], z_s_mu.shape[1]) + z_s_logvar - z_s_mu ** 2 - T.exp(z_s_logvar),
            axis=1).reshape((batch_size, 1)))
        l_obj = log_p_x_src_tgt + log_p_y + kl_anneal * kl_s

        return l_obj, log_p_x_src_tgt, kl_s


    def compile_train_l(self, input_sl, input_tl, y_tl, temp_anneal, kl_anneal, y_sl=None):
        batch_size = input_sl.shape[0]

        enc_src = self.encoder.build_encoder(input_sl)
        enc_tgt = self.encoder.build_encoder(input_tl)

        z_s_mu = self.z_mu(enc_src)
        z_s_logvar = self.z_logvar(enc_src)
        z_s = z_s_mu + T.exp(0.5 * z_s_logvar) * self.rng.normal(z_s_mu.shape, avg=0.0, std=1.0)
        z_t_mu = self.z_mu(enc_tgt)
        z_t_logvar = self.z_logvar(enc_tgt)
        z_t = z_t_mu + T.exp(0.5 * z_t_logvar) * self.rng.normal(z_t_mu.shape, avg=0.0, std=1.0)

        if not self.has_ly_src:
            q_y_x_unlabeled, logits = self.q_y_x(enc_src)
            y_sl = []
            for i in range(0, self.class_num):
                y_sl.append(gumbel_softmax(logits[i], temp_anneal, self.rng, True))
            # get the entropy of q(y|x)
            log_q_y_x_unlabeled = -self._get_prob_q_x_y(y_sl, q_y_x_unlabeled)

        dec_init_y_tgt, y_ctx_tgt, y_ctx_mask_tgt = self.y_dense_p(y_tl)
        dec_init_y_src, y_ctx_src, y_ctx_mask_src = self.y_dense_p(y_sl)
        # cross decode + self decode
        # source cross decode
        dec_init_state_tgt_to_src = self.p_h_yz(z_t, dec_init_y_src)
        log_p_x_tgt_src, _, att_probs_tgt_src = self.decoder.build_decoder(input_sl, context=y_ctx_src,
                                                                          c_mask=y_ctx_mask_src, fixed_context=z_t,
                                                                          provide_init_h=dec_init_state_tgt_to_src)
        log_p_x_tgt_src = T.mean(-log_p_x_tgt_src.reshape((batch_size, 1)))

        if not self.cross_only:
            # source self decode
            dec_init_state_src_to_src = self.p_h_yz(z_s, dec_init_y_src)
            log_p_x_src_src, _, att_probs_src_src = self.decoder.build_decoder(input_sl, context=y_ctx_src,
                                                                               c_mask=y_ctx_mask_src, fixed_context=z_s,
                                                                               provide_init_h=dec_init_state_src_to_src)
            log_p_x_src_src = T.mean(-log_p_x_src_src.reshape((batch_size, 1)))
            # target self decode
            dec_init_state_tgt_to_tgt = self.p_h_yz(z_t, dec_init_y_tgt)
            log_p_x_tgt_tgt, _, att_probs_tgt_tgt = self.decoder.build_decoder(input_tl, context=y_ctx_tgt,
                                                                              c_mask=y_ctx_mask_tgt, fixed_context=z_t,
                                                                              provide_init_h=dec_init_state_tgt_to_tgt)
            log_p_x_tgt_tgt = T.mean(-log_p_x_tgt_tgt.reshape((batch_size, 1)))


        # target cross decode
        dec_init_state_src_to_tgt = self.p_h_yz(z_s, dec_init_y_tgt)
        log_p_x_src_tgt, _, att_probs_src_tgt = self.decoder.build_decoder(input_tl, context=y_ctx_tgt,
                                                              c_mask=y_ctx_mask_tgt, fixed_context=z_s,
                                                              provide_init_h=dec_init_state_src_to_tgt)
        log_p_x_src_tgt = T.mean(-log_p_x_src_tgt.reshape((batch_size, 1)))

        if not self.cross_only:
            log_p_x_yz = log_p_x_src_src + log_p_x_tgt_src + log_p_x_src_tgt + log_p_x_tgt_tgt
        else:
            log_p_x_yz = log_p_x_tgt_src + log_p_x_src_tgt
        # priors
        log_p_y = self.p_y(y_tl)
        log_p_y += self.p_y(y_sl)
        log_p_y = T.mean(log_p_y)

        kl_s = T.mean(-0.5 * T.sum(
                tensor.alloc(1.0, z_s_mu.shape[0], z_s_mu.shape[1]) + z_s_logvar - z_s_mu ** 2 - T.exp(z_s_logvar),
                axis=1).reshape((batch_size, 1)))
        kl_t = T.mean(-0.5 * T.sum(
                tensor.alloc(1.0, z_t_mu.shape[0], z_t_mu.shape[1]) + z_t_logvar - z_t_mu ** 2 - T.exp(z_t_logvar),
                axis=1).reshape((batch_size, 1)))

        kl_st = kl_s + kl_t

        # classification error
        q_y_x_labeled, py = self.q_y_x(enc_tgt, labeled=True, Y=y_tl)

        if self.has_ly_src:
            q_y_x_labeled_src, _ = self.q_y_x(enc_src, labeled=True, Y=y_sl)
            q_y_x_labeled += q_y_x_labeled_src
        q_y_x_labeled = T.mean(q_y_x_labeled)

        if not self.has_ly_src:
            l_inner = log_p_x_yz + log_p_y + kl_anneal * kl_st + T.mean(log_q_y_x_unlabeled)
        else:
            l_inner = log_p_x_yz + log_p_y + kl_anneal * kl_st
        l_obj = l_inner + self.alpha * q_y_x_labeled

        recon_loss = log_p_x_yz

        return l_obj, q_y_x_labeled, l_inner, recon_loss, py, kl_st

    def compile_train_u(self, input_u, kl_anneal, temp_anneal):
        batch_size = input_u.shape[0]
        qx = self.encoder.build_encoder(input_u)
        q_y_x_unlabeled, logits = self.q_y_x(qx) # q_y_x_unlabeled is a list: each element is the probability (batch_size, #label_i)

        sampled_y = []
        for i in range(0, self.class_num):
            sampled_y.append(gumbel_softmax(logits[i], temp_anneal, self.rng, True))
        z_mu = self.z_mu(qx)
        z_logvar = self.z_logvar(qx)
        z = z_mu + T.exp(0.5 * z_logvar) * self.rng.normal(z_mu.shape, avg=0.0, std=1.0)

        dec_init_y, y_contxt_source, y_contxt_mask = self.y_dense_p(sampled_y)
        dec_init_state = self.p_h_yz(z, dec_init_y)

        log_p_x_yz, _, att_probs = self.decoder.build_decoder(input_u, context=y_contxt_source,
                                                              c_mask=y_contxt_mask, fixed_context=z,
                                                              provide_init_h=dec_init_state)
        log_p_x_yz = -log_p_x_yz.reshape((batch_size, 1))
        log_p_x_yz = T.mean(log_p_x_yz)
        log_p_y = self.p_y(sampled_y)  # already negative
        log_p_y = T.mean(log_p_y)
        if self.both_gaussian:
            kl_q_p = -0.5 * T.sum(
                tensor.alloc(1.0, z_mu.shape[0], z_mu.shape[1]) + z_logvar - z_mu ** 2 - T.exp(z_logvar),
                axis=1).reshape((batch_size, 1))
        else:
            log_p_z = -T.sum(standard_normal(z), axis=1).reshape((batch_size, 1))  ## negative???????
            log_q_z = T.sum(normal2(z_mu, z_logvar), axis=1).reshape((batch_size, 1))
            kl_q_p = log_p_z + log_q_z

        log_q_y_x = -self._get_prob_q_x_y(sampled_y, q_y_x_unlabeled)

        kl_q_p = T.mean(kl_q_p)

        u_inner = log_p_x_yz + log_p_y + kl_anneal * kl_q_p
        u_obj = u_inner + T.mean(log_q_y_x)

        return u_obj, u_inner, log_p_x_yz, q_y_x_unlabeled, kl_q_p, att_probs

    def compile_classifcation(self, x, y):
        qx = self.encoder.build_encoder(x)
        q_y_x, _ = self.q_y_x(qx, labeled=True, Y=y)
        loss = T.mean(q_y_x)
        return loss

    def _get_prob_q_x_y(self, uy, q_y_x_unlabeled):
        ## calculate entropy: q(y1)q(y2)q(y3)...log[q(y1)q(y2)q(y3)...]
        batch_size = q_y_x_unlabeled[0].shape[0]
        output = T.zeros((batch_size, 1))
        for i, y in enumerate(uy):
            # entropy = T.sum(q_y_x_unlabeled[i] * T.log(q_y_x_unlabeled[i]) * y, axis=)
            output += T.nnet.categorical_crossentropy(q_y_x_unlabeled[i], y).reshape((batch_size, 1))
        return output