import os
import os.path as osp
import pickle
import itertools as it
import argparse
import yaml
import time, datetime
import random
#random.seed(2333)

import numpy as np


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('yml_path', help='path to pkl_files')
    args = parser.parse_args()
    yml_file = open(args.yml_path, encoding='utf-8')
    param_dict = yaml.safe_load(yml_file)
    print(param_dict)
    for item in param_dict:
        parser.add_argument('--'+item, type=type(param_dict[item]), default=param_dict[item])
    args = parser.parse_args()
    return args


def time_count(func):
    def wrapper(*args):
        start = time.time()
        result = func(*args)
        print("cost time for {}: {:f}s".format(func.__name__, time.time()-start))
        return result
    return wrapper


# TP,FN,TN,FP
def cal_sensitivity(num_list):
    result_list = []
    for TP, FN, TN, FP in num_list:
        result_list.append(float(TP)/(TP+FN))
    return result_list


# TP,FN,TN,FP
def cal_specificity(num_list):
    result_list = []
    for TP, FN, TN, FP in num_list:
        result_list.append(float(TN)/(FP+TN))
    return result_list


def cal_accuracy(num_list):
    result_list = []
    for TP, FN, TN, FP in num_list:
        result_list.append(float(TP+TN)/(TP+TN+FP+FN))
    return result_list


def cal_exclude(num_list):
    result_list = []
    for TP, FN, TN, FP in num_list:
        result_list.append(float(TP)/(TP+TN+FP+FN))
    return result_list


def scan_pickle_file(directory_path):
    path = osp.join(directory_path)
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.split('.')[-1].lower() == 'pkl':
                file_list.append(osp.join(root, file))
    return file_list


def scan_pickle_sub_folder(pickle_file_directory):
    path = osp.join(pickle_file_directory)
    if not osp.isdir(path):
        raise Exception('No such pickle_file_directory')
    return os.listdir(pickle_file_directory)


def folder_check(class_order, class_folder):
    for s_class in class_order:
        if s_class not in class_folder:
            raise Exception("class {} not in folder, please check input folder".format(s_class))


def update_result(result, min_box_thresh, max_box_thresh):
    new_result = {}
    for k in result:
        label_boxes = np.array(result[k])

        if label_boxes.shape[0] > 0:
            label_boxes = label_boxes[np.logical_and(label_boxes[:, -1] > min_box_thresh, label_boxes[:, -1] <= max_box_thresh)]
            new_result[k] = label_boxes
        else:
            new_result[k] = result[k]
    return new_result


def split_data(fold_k, dataset):
    random.shuffle(dataset)
    num_data_per_set = [int(len(dataset)/fold_k)] * fold_k
    for i in range(len(dataset) % fold_k):
        num_data_per_set[i] += 1

    num_data_per_set.insert(0, 0)
    num_data_per_set = np.array(num_data_per_set)
    num_data_per_set_index = np.cumsum(num_data_per_set)

    new_data = [[] for _ in range(fold_k)]
    for i in range(fold_k):
        new_data[i].extend(dataset[num_data_per_set_index[i]:num_data_per_set_index[i+1]])

    return new_data


def get_fold_data(fold_k, input_path, class_order, feature_order, box_thresh_list, pos_neg=False, feat_type=['num']):
    class_folder = scan_pickle_sub_folder(input_path)
    folder_check(class_order, class_folder)

    class_file_list = []
    class_inds = []
    for class_index, s_class in enumerate(class_order):
        class_file_list.append(scan_pickle_file(osp.join(input_path, s_class)))
        class_inds.append(class_index)

    all_data = [[] for _ in range(fold_k)]
    for cls_index, class_file_list in zip(class_inds, class_file_list):
        current_cls_data = []
        for file_path in class_file_list:
            tmp_feats = []
            with open(file_path, 'rb') as f:
                result = pickle.load(f)
                if 'result' in result:
                    result = result['result']

            # TODO: move all if feat_type method to data feat func
            if 'overall' in feat_type:
                current_total_num = sum([len(result[k]) for k in feature_order])
                for s_feature in feature_order:
                    if current_total_num == 0:
                        tmp_feats.append(0.)
                    else:
                        tmp_feats.append(float(len(result[s_feature]) / current_total_num))

            for thresh_index in range(len(box_thresh_list)):
                current_result = update_result(result, box_thresh_list[thresh_index][0], box_thresh_list[thresh_index][1])

                if 'num' in feat_type:
                    for s_feature in feature_order:
                        tmp_feats.append(len(current_result[s_feature]))

                if 'sub_overall' in feat_type:
                    current_total_num = sum([len(current_result[k]) for k in feature_order])
                    for s_feature in feature_order:
                        if current_total_num == 0:
                            tmp_feats.append(0.)
                        else:
                            tmp_feats.append(float(len(current_result[s_feature])/current_total_num))

                if 'self' in feat_type:
                    for s_feature in feature_order:
                        if len(result[s_feature]) == 0:
                            tmp_feats.append(0.)
                        else:
                            tmp_feats.append(float(len(current_result[s_feature])/len(result[s_feature])))

            if pos_neg:
                tmp_feats.append(0 if cls_index == 0 else 1)
                class_list = ['neg', 'pos']
            else:
                tmp_feats.append(cls_index)
                class_list = class_order
            tmp_feats.append(file_path)
            current_cls_data.append(tmp_feats)


        # split data
        current_cls_data = split_data(fold_k, current_cls_data)

        for k in range(fold_k):
            all_data[k].extend(current_cls_data[k])


    return all_data, class_list


def get_params_comb(params_dict, max_comb=10):
    param_key = sorted(params_dict)
    all_param_comb = list(it.product(*(params_dict[n] for n in param_key)))
    max_comb = len(all_param_comb) if max_comb is None else min(max_comb, len(all_param_comb))
    random_param_comb = random.sample(all_param_comb, max_comb)
    comb_param_list = []
    for current_param_list in random_param_comb:
        current_param_dict = {k: p for k, p in zip(param_key, current_param_list)}
        comb_param_list.append(current_param_dict)
    return comb_param_list


def save_pkl(pkl, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(pkl, f)


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        content = pickle.load(f)
    return content


def get_eval_content(eval_dict, class_list):
    str_content = ''
    eval_names = sorted(eval_dict)
    for k in eval_names:
        str_content += '{}: \n'.format(k)
        for cls_name, value in zip(class_list, eval_dict[k]):
            str_content += '{}: {}\n'.format(cls_name, str(value))

    return str_content


def get_model_param_content(param_dict, class_list, feat_param):
    str_content = ''
    str_content += 'class: {}\n'.format(str(class_list))
    for tmp_dict in [param_dict, feat_param]:
        param_names = sorted(tmp_dict)
        for k in param_names:
            str_content += '{}: {}\n'.format(k, str(tmp_dict[k]))

    return str_content


def save_all_files(output_path, output_prefix, eval_dict, class_list, model_list, current_param_dict, feat_param):
    overall_content = ''
    date = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    folder_name = '{}_{}'.format(output_prefix, date)

    eval_str = get_eval_content(eval_dict, class_list)
    model_param_str = get_model_param_content(current_param_dict, class_list, feat_param)

    model_pkl = [model_list, class_list, feat_param]

    output_folder = os.path.join(output_path, folder_name)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # save write str content to file
    print(output_folder + ':\n')
    print(eval_str)
    overall_content += output_folder
    overall_content += eval_str
    for str_content, str_file in zip([eval_str, model_param_str], ['eval_{}.txt'.format(date),
                                                                   'params_{}.txt'.format(date)]):
        with open(os.path.join(output_folder, str_file), 'w') as f:
            f.write(str_content)

    # save model to file
    save_pkl(model_pkl, os.path.join(output_folder, 'model_{}.mlm'.format(date)))
    return overall_content


def train_test(all_data, output_path, class_list, feat_param, output_prefix='', add_data=None, params_dict=None, n_jobs=-1, max_param_comb_num=30):
    fold_k = len(all_data)

    val_data = [all_data[i] for i in range(fold_k)]

    train_data = [all_data[:i]+all_data[i+1:] for i in range(fold_k)]

    # combine each list data in train_data, and random shuffle
    for i in range(fold_k):
        tmp_list = []
        for k in range(len(train_data[i])):
            tmp_list.extend(train_data[i][k])
        random.shuffle(tmp_list)
        train_data[i] = tmp_list

    if add_data:
        for i in range(len(val_data)):
            val_data[i].extend(add_data[0])

    # get all training params
    comb_param_list = get_params_comb(params_dict, max_comb=max_param_comb_num)

    overall_content = ''
    for current_param_dict in comb_param_list:
        print("training start for params: \n")
        print(current_param_dict)
        start = time.time()

        model_rec_list = []
        eval_dict = {}

        # TP,FN,TN,FP
        num_count_per_label_list = [[0, 0, 0, 0] for _ in range(len(class_list))]

        for single_train_data, single_val_data in zip(train_data, val_data):
            single_train_data = np.array(single_train_data)
            single_val_data = np.array(single_val_data)

            X_train, Y_train = np.array(single_train_data[:, :-2], dtype=float), np.array(single_train_data[:, -2], dtype=int)

            X_val, Y_val = np.array(single_val_data[:, :-2], dtype=float), np.array(single_val_data[:, -2], dtype=int)


            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(**current_param_dict, n_jobs=n_jobs).fit(X_train, Y_train)

            model_rec_list.append(clf)

            pre_result = clf.predict(X_val)


            # tp, fn per label
            for cls_index, _ in enumerate(class_list):
                P = Y_val[Y_val == cls_index].shape[0]
                N = Y_val.shape[0] - P

                correct_result = pre_result[pre_result == Y_val]
                TP = correct_result[correct_result == cls_index].shape[0]
                FN = P - TP

                TN = correct_result[correct_result != cls_index].shape[0]
                FP = N - TN

                num_count_per_label_list[cls_index][0] += TP
                num_count_per_label_list[cls_index][1] += FN
                num_count_per_label_list[cls_index][2] += TN
                num_count_per_label_list[cls_index][3] += FP

        sen_list = cal_sensitivity(num_count_per_label_list)
        eval_dict['sen'] = sen_list
        spec_list = cal_specificity(num_count_per_label_list)
        eval_dict['spec'] = spec_list
        acc_list = cal_accuracy(num_count_per_label_list)
        eval_dict['acc'] = acc_list
        exc_list = cal_exclude(num_count_per_label_list)
        eval_dict['exc'] = exc_list
        print("cost time: {}".format(time.time()-start))
        tmp_content = save_all_files(output_path, output_prefix, eval_dict, class_list, model_rec_list, current_param_dict, feat_param)

        with open(os.path.join(output_path, 'overall.txt'), 'a') as f:
            f.write(tmp_content)


def main():
    # get all params from yaml file
    args = parse_arg()

    # create dict for random dataset params
    # TODO auto generate params dict ?
    feature_params_dict = args.dataset_params

    # parse args, find random params
    comb_feature_param_list = get_params_comb(feature_params_dict, max_comb=args.max_dataset_param_comb_num)

    # create dict for random training params
    train_params_dict = args.random_forest_params

    # loop for random param comb
    for feat_param in comb_feature_param_list:
        print('current feat param:')
        print(feat_param)
        all_data, class_list = get_fold_data(args.fold_k, args.input_path, args.class_order, **feat_param)
        if args.add_input_path:
            add_data, _ = get_fold_data(1, args.add_input_path, args.class_order, **feat_param)
        else:
            add_data = None

        # create dict fro random prams in
        train_test(all_data, args.output_path, class_list, feat_param, output_prefix=args.output_prefix, add_data=add_data, params_dict=train_params_dict,
                   n_jobs=args.n_jobs, max_param_comb_num=args.max_param_comb_num)


if __name__ == '__main__':
    main()
