__author__ = 'chuntingzhou'
import sys
import os
import theano

sys.path.append('../')
from data import data_sup
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import logging
import time
import numpy as np
from semi_models.morph_att import SSL_VAE
np.set_printoptions(threshold=np.inf)
from emolga.utils.generic_utils import init_logging
from semi_models.utils import *
import argparse
import codecs


def write_config(config, filename):
    with open(filename, 'w') as fout:
        for k, v in config.iteritems():
            fout.write("%s %s\n" % (k, str(v)))


def write_prediction_to_file(config, ll):
    with codecs.open(os.path.join(config['dump_dir'], 'best_prediction.out'), 'w', "utf-8") as fout:
        for l in ll:
            fout.write(" ".join(l) + '\n')
    with codecs.open(os.path.join(config['dump_dir'], 'wrong_prediction.out'), 'w', "utf-8") as fout:
        for l in ll:
            if l[2] != l[1]:
                fout.write(" ".join(l) + '\n')


def main(config):
    debug = False
    param_print = ""
    for ind in config["index_options"]:
        param_print += ind + "-" + str(config[ind]) + "-"

    if not config["test"]:
        config['dump_dir'] = "../obj/twoz_%s_time-%s/" % (param_print, time.strftime("%y.%m.%d_%H.%M.%S"))
        if debug:
            config['dump_dir'] = "../obj/debug/"
        else:
            os.mkdir(config['dump_dir'])

    init_logging(os.path.join(config['dump_dir'], "run.log"), logging.INFO)

    epc = 1
    epochs = config["epochs"]
    batch_size = 20
    test_batch_size = 100
    dispfreq = 100
    genfreq = 100
    genAccFreq = 2000
    savefreq = 1000
    disppred = False
    prune_train_data = False
    update_temp = 2000
    check_att = False

    logging.info("Process data.....")
    voc_size, class_num, label_list, ix_to_char, ix_to_label, ix_to_tag, x_test_src, x_test_tgt, y_test_tgt = data_sup.preprocess(config["add_uns"])
    logging.info("Done.")

    logging.info(ix_to_char)

    test_size = len(x_test_src)

    has_ly_src = config['has_ly_src']
    save = True
    config['enc_voc_size'] = voc_size
    config['dec_voc_size'] = voc_size
    config['class_num'] = class_num
    config['label_list'] = label_list
    config['batch_size'] = batch_size

    ux = np.array(data_sup.ux)
    uy = np.array(data_sup.uy)
    lx_src = np.array(data_sup.lx_src)
    ly_src = np.array(data_sup.ly_src)
    lx_tgt = np.array(data_sup.lx_tgt)
    ly_tgt = np.array(data_sup.ly_tgt)
    ux = ux[:config['ul_num']]
    logging.info("length of ux and uy: %d, %d", len(ux), len(uy))
    logging.info("Total unlabeled training samples: %d", len(ux))
    logging.info("Total labeled training samples before pruning: %d", len(lx_src))
    larger = "u" if len(ux) > len(lx_src) else "l"
    assert len(lx_src) == len(ly_tgt) and len(lx_tgt) == len(ly_tgt)
    labeled = 12000
    unlabeled = len(ux)

    if prune_train_data:
        inds = np.random.permutation(len(lx_src))
        inds = inds[:labeled]
        lx_src = lx_src[inds]
        if config['has_ly_src']:
            ly_src = ly_src[inds]
        lx_tgt = lx_tgt[inds]
        ly_tgt = ly_tgt[inds]

    labeled = len(lx_src)
    logging.info("Total labeled training samples: %d", len(lx_src))

    l_batches = data_sup.get_batches(labeled, batch_size)
    u_batches = data_sup.get_batches(unlabeled, batch_size)
    l_cur_batch = 0
    u_cur_batch = 0
    rng = RandomStreams(1234)

    if config['test']:
        genAccFreq = 100
        if config['loadfrom'] is None:
            assert False, "Load Path not Provided"    

    ssl_vae = SSL_VAE(rng, config)
    logging.info("Start building and compiling the model....")
    ssl_vae.compile_train()
    logging.info("Compile done.")
    logging.info("Start compile generating model...")
    ssl_vae.compile_gen()
    logging.info("Compile generation done.")

    begin_time = time.time()

    debug_bs = 10
    cur = 0

    write_config(config, os.path.join(config['dump_dir'], 'exp.config'))

    def compute_acc(q_y_x, y, batch_size):
        # q_y_x: list of [(batch_size, #label)]
        # y: list of [(batch_size, #label)] 1-of-k
        # logging.info("check probs:", q_y_x)
        prediction = [np.argmax(q_y_x[i], axis=1) for i in range(0, class_num)]
        true_label = [np.argmax(y[i], axis=1) for i in range(0, class_num)]
        prediction = np.array(prediction)
        true_label = np.array(true_label)
        # logging.info("size of true label and prediction label: ", true_label.shape, prediction.shape)
        acc = (prediction == true_label)  # (class_num, batch_size)
        joint_acc = np.sum(acc, axis=0)
        ground = np.ones(batch_size) * class_num
        joint_acc = np.sum(joint_acc == ground) * 1.0 / batch_size
        separate_acc = np.sum(acc, axis=1) * 1.0 / batch_size
        avg_acc = sum(separate_acc) * 1.0 / class_num
        return joint_acc, separate_acc, prediction, true_label, avg_acc

    temp = 1.0
    update_ind = 0
    valid_history = []
    patience = 30
    bad_counter = 0
    logging.info(config)

    while epc < epochs:
        logging.info('Epoch %d starts\n', epc)
        tot_nb_examples = 0
        tot_loss = 0.0
        tot_obj_l = 0.0
        tot_obj_u = 0.0
        tot_obj_pre_l = 0.0
        tot_obj_acc_l = 0.0
        tot_obj_acc_u = 0.0

        while True:
            if debug:
                cur += 1
                if cur > debug_bs:
                    break

            if l_cur_batch >= len(l_batches):
                l_cur_batch = 0
                inds = np.random.permutation(len(lx_src))
                lx_src = lx_src[inds]
                if config['has_ly_src']:
                    ly_src = ly_src[inds]
                lx_tgt = lx_tgt[inds]
                ly_tgt = ly_tgt[inds]
                if larger == "l":
                    break
            if u_cur_batch >= len(u_batches):
                u_cur_batch = 0
                inds = np.random.permutation(len(ux))
                ux = ux[inds]
                # uy = uy[inds]
                if larger == "u":
                    break

            lstart, lend = l_batches[l_cur_batch][0], l_batches[l_cur_batch][1]
            lx_src_batch = lx_src[lstart:lend]
            if config['has_ly_src']:
                ly_src_batch = ly_src[lstart:lend]
            lx_tgt_batch = lx_tgt[lstart:lend]
            ly_tgt_batch = ly_tgt[lstart:lend]

            ustart, uend = u_batches[u_cur_batch][0], u_batches[u_cur_batch][1]
            uxbatch = ux[ustart:uend]
            # uybatch = uy[ustart:uend]

            l_cur_batch += 1
            u_cur_batch += 1

            if has_ly_src:
                lx_src_batch, lx_src_mask, ly_src_batch = data_sup.prepare_xy_batch(lx_src_batch, ly_src_batch,
                                                                                    label_list)
            else:
                lx_src_batch, lx_src_mask = data_sup.prepare_x_batch(lx_src_batch)
            lx_tgt_batch, lx_tgt_mask, ly_tgt_batch = data_sup.prepare_xy_batch(lx_tgt_batch, ly_tgt_batch, label_list)
            uxbatch, uxmask = data_sup.prepare_x_batch(uxbatch)

            if config["disable_kl"]:
                kl_w = 0.0
            else:
                kl_w = get_kl_weight(update_ind, config['kl_thres'], config['kl_rate'])

            if update_ind % update_temp == 0:
                temp = get_temp(update_ind)

            if config["sl_anneal"]:
                dt_sl = get_sl_weight(update_ind)
            else:
                dt_sl = config["dt_sl"]

            if has_ly_src:
                inputs = [lx_src_batch] + ly_src_batch + [lx_tgt_batch] + ly_tgt_batch + [uxbatch] + [kl_w] + [temp] + [dt_sl]
            else:
                inputs = [lx_src_batch] + [lx_tgt_batch] + ly_tgt_batch + [uxbatch] + [kl_w] + [temp] + [dt_sl]

            outputs = ssl_vae.train_(*inputs)

            loss = outputs[0]
            obj_l = outputs[1]
            q_y_x_loss_l = outputs[2]
            obj_u = outputs[3]
            recon_l = outputs[4]
            recon_u = outputs[5]
            kl_l = outputs[6]
            kl_u = outputs[7]
            q_y_x_l = outputs[8:8 + class_num]
            q_y_x_u = outputs[-class_num:]

            accl, sep_acc_l, pred_l, yl, avg_acc_l = compute_acc(q_y_x_l, ly_tgt_batch, len(lx_src_batch))

            if update_ind % dispfreq == 0:
                logging.info('Update %d, kl annealing weight = %.3f', update_ind, kl_w)
                logging.info('Update %d, gumbel softmax temperature = %.3f', update_ind, temp)
                logging.info('Update %d, loss = %.3f, labeled loss = %.3f, '
                             'unlabeled loss = %.3f, labeled pred loss = %f, '
                             'labeled pred acc = %f, '
                             'labeled recon loss = %f, unlabeled recon loss = %f, '
                             'labeled kl loss = %f, unlabeled kl loss = %f, ',
                             # 'labeled kl source and target = %f',
                             update_ind, loss, obj_l, obj_u, q_y_x_loss_l, accl, recon_l, recon_u, kl_l, kl_u
                             )

            if update_ind % genfreq == 0:
                # let's generate something different!
                # first evaluate the analysis ability:
                test_batches = data_sup.get_batches(len(x_test_src), test_batch_size)
                test_obj_l = 0.0
                test_q_y_x_loss = 0.0
                test_acc = 0.0
                test_avg_acc = 0.0
                for i in range(0, len(test_batches)):
                    starts = test_batches[i][0]
                    ends = test_batches[i][1]
                    x_test_tgt_batch = x_test_tgt[starts:ends]
                    y_test_tgt_batch = y_test_tgt[starts:ends]
  
                    test_x_tgt, _, test_y_tgt = data_sup.prepare_xy_batch(x_test_tgt_batch, y_test_tgt_batch,
                                                                          label_list)
                    # test_x_tgt, _, test_y_tgt = data_sup.prepare_xy_batch(x_tgt_batch, y_tgt_batch, label_list)
                    test_inputs = [test_x_tgt] + test_y_tgt + [kl_w]
                    test_outputs = ssl_vae.test_pred_(*test_inputs)
                    test_q_y_x_loss += test_outputs[0]
                    test_q_y_x = test_outputs[-class_num:]
                    cur_bs = len(test_x_tgt)
                    test_joint_acc, test_separate_acc, test_prediction, test_true_label, test_avg = compute_acc(
                        test_q_y_x, test_y_tgt, cur_bs)
                    test_acc += test_joint_acc
                    test_avg_acc += test_avg
                test_bs = len(test_batches)
                test_obj_l /= test_bs
                test_acc /= test_bs
                test_avg_acc /= test_bs
                logging.info(
                    '********* Test *********: test labeled data loss = %.3f, test acc = %f, test avg acc = %f.',
                    test_obj_l, test_acc, test_avg_acc)

                inds = np.random.permutation(test_size)
                inds = inds[:10]
                corr = 0
                corr_gen = 0
                for i in inds:
                    x, _, y = data_sup.prepare_xy_batch([x_test_src[i]], [y_test_tgt[i]], label_list)
                    test_inputs = [x] + y
                    dec_init_h, cxt, cxt_mask, z = ssl_vae.f_get_dec_inith_cxt(*test_inputs)
                    fixed_cxt = z
                    sample, score = ssl_vae.decoder.get_sample(cxt, cxt_mask, fixed_cxt, dec_init_h,
                                                               k=5, maxlen=20, stochastic=False)

                    src_word = [ix_to_char[c] for c in x_test_src[i]]
                    tgt_word = [ix_to_char[c] for c in x_test_tgt[i]]
                    logging.info("******INFLECTION GENARATION*********")
                    logging.info("Source word: %s", u" ".join(src_word))
                    logging.info("Target word: %s", u" ".join(tgt_word))
                    # score = score / np.array([len(s) for s in sample])
                    word = sample[np.array(score).argmin()]
                    gen_word = []
                    for c in word:
                        if c == 0:
                            break
                        else:
                            gen_word.append(ix_to_char[c])
                    logging.info("Sample word: %s", u" ".join(gen_word))
                    logging.info("***********************************")
                    if ''.join(tgt_word) == ''.join(gen_word):
                        corr_gen += 1

                    ######################################################################################
                    x_at, _, y_at = data_sup.prepare_xy_batch([x_test_tgt[i]], [y_test_tgt[i]], label_list)
                    test_inputs_at = [x_at] + y_at
                    dec_init_at, cxt_at, cxt_mask_at, fixed_cxt_at = ssl_vae.f_get_dec_inith_cxt(*test_inputs_at)
                    sample_at, score_at = ssl_vae.decoder.get_sample(cxt_at, cxt_mask_at, fixed_cxt_at, dec_init_at,
                                                                     k=5, maxlen=50, stochastic=False)
                    tgt_word = [ix_to_char[c] for c in x_test_tgt[i]]
                    logging.info("*******AUTOENCODER************")
                    logging.info("Origin word: %s", u" ".join(tgt_word))
                    word = sample_at[np.array(score_at).argmin()]
                    gen_word = []
                    for c in word:
                        if c == 0:
                            break
                        else:
                            gen_word.append(ix_to_char[c])
                    logging.info("Sample word: %s", u" ".join(gen_word))
                    logging.info("***********************************")
                    if ''.join(tgt_word) == ''.join(gen_word):
                        corr += 1
                logging.info(
                    "%d words out of 10 are correctly reconstructed, %d words out of 10 are correctly generated!", corr,
                    corr_gen)

            update_ind += 1

            predict_list = []
            if update_ind % genAccFreq == 0 and update_ind >= config["start_val"]:
                corr = 0
                for i in range(0, test_size):
                    x, _, y = data_sup.prepare_xy_batch([x_test_src[i]], [y_test_tgt[i]], label_list)
                    test_inputs = [x] + y
                    dec_init_h, cxt, cxt_mask, z = ssl_vae.f_get_dec_inith_cxt(*test_inputs)
                    fixed_cxt = z
                    sample, score = ssl_vae.decoder.get_sample(cxt, cxt_mask, fixed_cxt, dec_init_h,
                                                               config['sample_beam'], maxlen=50, stochastic=False)

                    src_word = [ix_to_char[c] for c in x_test_src[i]]
                    tgt_word = [ix_to_char[c] for c in x_test_tgt[i]]
                    # score = score / np.array([len(s) for s in sample])
                    word = sample[np.array(score).argmin()]
                    gen_word = []
                    for c in word:
                        if c == 0:
                            break
                        else:
                            gen_word.append(ix_to_char[c])
                    if u''.join(tgt_word) == u''.join(gen_word):
                        corr += 1

                    ll = [u''.join(src_word), u''.join(tgt_word), u''.join(gen_word)]
                    predict_list.append(ll)

                    if i % 1000 == 0:
                        logging.info('Processed %d test data.', i)
                logging.info("######### Accuracy of generation on test data: %f ################",
                             corr * 1.0 / test_size)
                gen_acc = corr * 1.0 / test_size
                if len(valid_history) == 0 or gen_acc >= max(valid_history):
                    bad_counter = 0
                    write_prediction_to_file(config, predict_list)
                    logging.info("Saving params to %s", config['dump_dir'])
                    ssl_vae.save(os.path.join(config['dump_dir'], 'params.pkl'))
                else:
                    bad_counter += 1
                    if bad_counter > patience:
                        logging.info("Early Stop!")
                        exit(0)
                valid_history.append(gen_acc)

            tot_nb_examples += 1
            tot_loss += loss
            tot_obj_l += obj_l
            tot_obj_u += obj_u
            tot_obj_pre_l += q_y_x_loss_l
            tot_obj_acc_l += accl
            
        epc += 1
        logging.info('[Epoch %d] cumulative loss = %.3f, avg. labeled loss = %.3f, '
                     'avg. labeled pred loss = %f,  avg. unlabeled loss = %.3f, '
                     'avg. labeled pred acc = %f,  (took %ds)',
                     epc,
                     tot_loss / tot_nb_examples,
                     tot_obj_l / tot_nb_examples,
                     tot_obj_u / tot_nb_examples,
                     tot_obj_pre_l / tot_nb_examples,
                     tot_obj_acc_l / tot_nb_examples,
                     time.time() - begin_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bidirectional", action="store_true", default=False)
    parser.add_argument("-y_dim", action="store", type=int, default=200)
    parser.add_argument("-z_dim", action="store", type=int, default=100)
    parser.add_argument("-cross_only", action="store_true", default=False)
    parser.add_argument("-disable_kl", action="store_true", default=False)
    parser.add_argument("-worddrop", action="store", type=float, default=0.3)
    parser.add_argument("-share_emb", action="store_false", default=True)
    parser.add_argument("-lang", action="store", type=str, default="arabic")
    parser.add_argument("-addsup", action="store_true", default=False)
    parser.add_argument("-kl_st", action="store_true", default=False)
    parser.add_argument("-kl_rate", action="store", type=float, default=150000.0)
    parser.add_argument("-kl_thres", action="store", type=float, default=0.4)
    parser.add_argument("-start_val", action="store", type=int, default=24000)
    parser.add_argument("-only_supervise", action="store_true", default=False)
    parser.add_argument("-dt_uns", action="store", type=float, default=0.7)
    parser.add_argument("-epochs", action="store", type=int, default=80)
    parser.add_argument("-reload", action="store_true", default=False)
    parser.add_argument("-loadfrom", action="store", type=str, default=None)
    parser.add_argument("-optimizer", action="store", type=str, default="adadelta")
    parser.add_argument("-withcxt", action="store_true", default=False)
    parser.add_argument("-hid_dim", action="store", type=int, default=256)
    parser.add_argument("-only_ul", action="store_true", default=False)
    parser.add_argument("-sl_anneal", action="store_true", default=False)
    parser.add_argument("-dt_sl", action="store", type=float, default=1.0)
    parser.add_argument("-alpha", action="store", type=float, default=1.0)
    parser.add_argument("-add_uns", action="store", type=float, default=0.2)
    parser.add_argument("-ul_num", action="store", type=float, default=10000)
    args = parser.parse_args()

    config = {}
    config["ul_num"] = args.ul_num
    config["add_uns"] = args.add_uns
    config['pure_sup'] = False
    config["epochs"] = args.epochs
    config["dt_uns"] = args.dt_uns
    config["dt_sl"] = args.dt_sl
    config["sl_anneal"] = args.sl_anneal
    config["only_sup"] = args.only_supervise
    config['start_val'] = args.start_val
    config['kl_rate'] = args.kl_rate
    config['kl_thres'] = args.kl_thres
    config['withcxt'] = args.withcxt
    config['only_ul'] = args.only_ul
    config['test'] = False
    config['index_options'] = ['word_dropout', "y_dense_p_dim", "lang", "cross_only", "only_sup", "kl_thres", "dt_uns", "add_uns"]
    config['lang'] = args.lang
    data_sup.lang = args.lang
    print "language:", data_sup.lang
    config['reload'] = args.reload
    config['loadfrom'] = args.loadfrom
    config['has_ly_src'] = False
    config['cross_only'] = args.cross_only
    config["disable_kl"] = args.disable_kl
    config['alpha'] = args.alpha
    config['use_input'] = True
    config['both_gaussian'] = True
    # config['input_dim'] = 40
    config['dropout'] = 0.0
    config['word_dropout'] = args.worddrop
    # config['temperature'] = 0.1
    # config['sample_size'] = 30
    config['kl_st'] = args.kl_st
    config['activation_dense'] = 'tanh'
    config['init_dense'] = 'glorot_normal'

    config['enc_embedd_dim'] = 300  # 500 -> 300
    config['enc_hidden_dim'] = args.hid_dim
    # config['enc_contxt_dim'] = config['enc_hidden_dim']
    config['bidirectional'] = args.bidirectional  # change here
    config['shared_embed'] = args.share_emb
    # config['dec_embedd_dim']

    config['q_z_dim'] = args.z_dim  # z dim
    config['q_z_x_hidden_dim'] = None  # [enc_context_dim,128]
    config['y_dense_p_dim'] = args.y_dim  # y dim
    config['provide_init_h'] = True
    config['bias_code'] = False
    config['dec_embedd_dim'] = config['enc_embedd_dim']  # 128 -> 500
    config['dec_hidden_dim'] = config['enc_hidden_dim']
    # config['dec_contxt_dim'] = config['y_dense_p_dim'] + config['dec_hidden_dim']
    config['dec_contxt_dim'] = config['y_dense_p_dim'] + config['q_z_dim']

    config['add_sup'] = args.addsup
    config['deep_out'] = False
    # config['output_dim']
    # config['deep_out_activ']

    config['sample_stoch'] = False
    config['sample_beam'] = 8
    config['sample_argmax'] = False

    config['bigram_predict'] = False
    config['context_predict'] = True
    config['leaky_predict'] = False
    config['optimizer'] = args.optimizer

    main(config)
