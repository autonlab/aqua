import numpy as np
import random
import torch

from aqua.models.cleaning_modules.simifeat.hoc import *
import aqua.models.cleaning_modules.simifeat.global_var as global_var

def data_transform(record, noise_or_not, sel_noisy):
    # assert noise_or_not is not None
    total_len = sum([len(a) for a in record])
    origin_trans = torch.zeros(total_len, record[0][0]['feature'].shape[0])
    origin_label = torch.zeros(total_len).long()
    noise_or_not_reorder = np.empty(total_len, dtype=bool)
    index_rec = np.zeros(total_len, dtype=int)
    cnt, lb = 0, 0
    sel_noisy = np.array(sel_noisy)
    noisy_prior = np.zeros(len(record))

    for item in record:
        for i in item:
            # if i['index'] not in sel_noisy:
            origin_trans[cnt] = i['feature']
            origin_label[cnt] = lb
            noise_or_not_reorder[cnt] = noise_or_not[i['index']] if noise_or_not is not None else False
            index_rec[cnt] = i['index']
            cnt += 1 - np.sum(sel_noisy == i['index'].item())
            # print(cnt)
        noisy_prior[lb] = cnt - np.sum(noisy_prior)
        lb += 1
    data_set = {'feature': origin_trans[:cnt], 'noisy_label': origin_label[:cnt],
                'noise_or_not': noise_or_not_reorder[:cnt], 'index': index_rec[:cnt]}
    return data_set, noisy_prior / cnt


def get_knn_acc_all_class(args, data_set, k=10, noise_prior=None, sel_noisy=None, thre_noise_rate=0.5, thre_true=None):
    # Build Feature Clusters --------------------------------------
    KINDS = args.num_classes

    all_point_cnt = data_set['feature'].shape[0]
    # global
    sample = np.random.choice(np.arange(data_set['feature'].shape[0]), all_point_cnt, replace=False)
    # final_feat, noisy_label = get_feat_clusters(data_set, sample)
    final_feat = data_set['feature'][sample]
    noisy_label = data_set['noisy_label'][sample]
    noise_or_not_sample = data_set['noise_or_not'][sample]
    sel_idx = data_set['index'][sample]
    knn_labels_cnt = count_knn_distribution(args, final_feat, noisy_label, all_point_cnt, k=k, norm='l2')


    method = 'ce'
    # time_score = time.time()
    score = get_score(knn_labels_cnt, noisy_label, k=k, method=method, prior=noise_prior)  # method = ['cores', 'peer']
    # print(f'time for get_score is {time.time()-time_score}')
    score_np = score.cpu().numpy()

    if args.method == 'mv':
        # test majority voting
        logging.info(f'Use MV')
        label_pred = np.argmax(knn_labels_cnt, axis=1).reshape(-1)
        sel_noisy += (sel_idx[label_pred != noisy_label]).tolist()
    elif args.method == 'rank1':
        logging.info(f'Use rank1')
        logging.info(f'Tii offset is {args.Tii_offset}')
        # fig=plt.figure(figsize=(15,4))
        for sel_class in range(KINDS):
            thre_noise_rate_per_class = 1 - min(args.Tii_offset * thre_noise_rate[sel_class][sel_class], 1.0)
            if thre_noise_rate_per_class >= 1.0:
                thre_noise_rate_per_class = 0.95
            elif thre_noise_rate_per_class <= 0.0:
                thre_noise_rate_per_class = 0.05
            sel_labels = (noisy_label.cpu().numpy() == sel_class)
            thre = np.percentile(score_np[sel_labels], 100 * (1 - thre_noise_rate_per_class))

            indicator_all_tail = (score_np >= thre) * (sel_labels)
            sel_noisy += sel_idx[indicator_all_tail].tolist()
    else:
        raise NameError('Undefined method')

    return sel_noisy

    # plot_score(score, name = f'{args.noise_type}_{args.noise_rate}_{k}_{method}', noise_or_not_sample = noise_or_not_sample, thre_noise_rate = thre_noise_rate, sel_class = sel_class)

    # method = 'avg'
    # score = get_score(knn_labels_cnt, noisy_label, k = k, method = method, prior = noise_prior) # method = ['cores', 'peer']
    # plot_score(score, name = f'{args.noise_type}_{args.noise_rate}_{k}_{method}', noise_or_not_sample = noise_or_not_sample)
    # method = 'new'
    # score = get_score(knn_labels_cnt, noisy_label, k = k, method = method, prior = noise_prior) # method = ['cores', 'peer']
    # plot_score(score, name = f'{args.noise_type}_{args.noise_rate}_{k}_{method}', noise_or_not_sample = noise_or_not_sample)
    # exit()


def get_T_global_min_new(args, data_set, max_step=501, T0=None, p0=None, lr=0.1, NumTest=50, all_point_cnt=15000):


    # Build Feature Clusters --------------------------------------
    KINDS = args.num_classes
    # NumTest = 50
    all_point_cnt = args.cnt
    logging.info(f'Use {all_point_cnt} in each round. Total rounds {NumTest}.')

    p_estimate = [[] for _ in range(3)]
    p_estimate[0] = torch.zeros(KINDS)
    p_estimate[1] = torch.zeros(KINDS, KINDS)
    p_estimate[2] = torch.zeros(KINDS, KINDS, KINDS)
    # p_estimate_rec = torch.zeros(NumTest, 3)
    for idx in range(NumTest):
        # print(idx, flush=True)
        # global
        sample = np.random.choice(range(data_set['feature'].shape[0]), all_point_cnt, replace=False)
        # final_feat, noisy_label = get_feat_clusters(data_set, sample)
        final_feat = data_set['feature'][sample]
        noisy_label = data_set['noisy_label'][sample]
        cnt_y_3 = count_y(KINDS, final_feat, noisy_label, all_point_cnt)
        for i in range(3):
            cnt_y_3[i] /= all_point_cnt
            p_estimate[i] = p_estimate[i] + cnt_y_3[i] if idx != 0 else cnt_y_3[i]

    for j in range(3):
        p_estimate[j] = p_estimate[j] / NumTest

    #args.device = set_device()
    loss_min, E_calc, P_calc, _ = calc_func(KINDS, p_estimate, False, args.device, max_step, T0, p0, lr=lr)
    E_calc = E_calc.cpu().numpy()
    P_calc = P_calc.cpu().numpy()
    return E_calc, P_calc


# def error(T, T_true):
#     error = np.sum(np.abs(T - T_true)) / np.sum(np.abs(T_true))
#     return error


def noniterate_detection(config, record, train_dataset, sel_noisy=[]):

    T_given_noisy_true = None
    T_given_noisy = None


    # non-iterate
    # sel_noisy = []
    data_set, noisy_prior = data_transform(record, train_dataset.noise_or_not, sel_noisy)
    # print(data_set['noisy_label'])
    if config.method == 'rank1':
        T_init = global_var.get_value('T_init')
        p_init = global_var.get_value('p_init')

        # print(f'T_init is {T_init}')
        T, p = get_T_global_min_new(config, data_set=data_set, max_step=config.max_iter if T_init is None else 20,
                                    lr=0.1 if T_init is None else 0.01, NumTest=config.G, T0=T_init, p0=p_init)


        T_given_noisy = T * p / noisy_prior
        logging.info("T given noisy:")
        logging.info(np.round(T_given_noisy, 2))
        # add randomness
        for i in range(T.shape[0]):
            T_given_noisy[i][i] += np.random.uniform(low=-0.05, high=0.05)


    sel_noisy = get_knn_acc_all_class(config, data_set, k=config.k, noise_prior=noisy_prior, sel_noisy=sel_noisy,
                                      thre_noise_rate=T_given_noisy, thre_true=T_given_noisy_true)

    sel_noisy = np.array(sel_noisy)
    sel_clean = np.array(list(set(data_set['index'].tolist()) ^ set(sel_noisy)))

    logging.debug(f"Sel_noisy: {sel_noisy}")
    logging.debug(f"Sel_noisy shape: {sel_noisy.shape}")
    logging.debug(f"Sel_noisy type: {sel_noisy.dtype}")

    if sel_noisy.shape[0] == 0:
        sel_noisy = np.array(sel_noisy, dtype=int)
    noisy_in_sel_noisy = np.sum(train_dataset.noise_or_not[sel_noisy]) / sel_noisy.shape[0]
    precision_noisy = noisy_in_sel_noisy
    recall_noisy = np.sum(train_dataset.noise_or_not[sel_noisy]) / np.sum(train_dataset.noise_or_not)


    logging.debug(f'[noisy] precision: {precision_noisy}')
    logging.debug(f'[noisy] recall: {recall_noisy}')
    logging.debug(f'[noisy] F1-score: {2.0 * precision_noisy * recall_noisy / (precision_noisy + recall_noisy)}')

    return sel_noisy, sel_clean, data_set['index']