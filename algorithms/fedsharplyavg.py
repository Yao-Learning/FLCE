import copy
import numpy as np
import itertools
import fmodule
import time
from datetime import datetime


import torch
import torch.nn as nn

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs
from utils import shapley
from utils import powersettool
from utils import time_predict

from tools import construct_dataloaders
from tools import construct_optimizer

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from utils import time_diff
from datetime import datetime
import warnings
import time
import random


class FedSharplyAvg():
    def __init__(
        self, csets, gset, model, args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.initmodel = copy.deepcopy(model)
        self.args = args
        self.history_sv = []

        self.clients = list(csets.keys())

        self.start_time = time.time()

        self.k_idx = 0


        # construct dataloaders
        self.train_loaders, self.test_loaders, self.glo_test_loader = \
            construct_dataloaders(
                self.clients, self.csets, self.gset, self.args
            )

        self.logs = {
            "ROUNDS": [],
            "LOSSES": [],
            "GLO_TACCS": [],
            # "LOCAL_TACCS": [],
            "GLO_precision": [],
            "GLO_recall": [],
            "GLO_f1": [],
            # "GLO_auc": []
        }

    def train(self):
        client_total_time = 0
        server_total_time = 0
        start_time = time.time()
        print('#################开始训练时间为{}##################'.format(datetime.fromtimestamp(start_time)))

        # Training
        for r in range(1, self.args.max_round + 1):
            if r == 2:
                time_predict(start_time, self.args.max_round)

            n_sam_clients = int(self.args.c_ratio * len(self.clients))

            clients_idx_list = list(range(50))
            sam_clients = clients_idx_list[self.k_idx:self.k_idx + 5]

            if self.k_idx < 45:
                self.k_idx += 5
            else:
                self.k_idx = 0

            remaining_numbers = [num for num in clients_idx_list if num not in sam_clients]
            random_samples = random.sample(remaining_numbers, 5)
            sam_clients.extend(random_samples)

            # sam_clients = np.random.choice(
            #     self.clients, n_sam_clients, replace=False
            # )

            local_trainData = {}   #获取本地数量

            for client in sam_clients:
                local_dataNum = len(self.csets[client][0])
                local_trainData[client] = local_dataNum

            local_models = {}

            avg_loss = Averager()
            all_per_accs = []

            clients_start_time = time.time()
            print('clients', sam_clients)
            for client in sam_clients:
                local_model, per_accs, loss = self.update_local(
                    r=r,
                    model=copy.deepcopy(self.model),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                )

                local_models[client] = copy.deepcopy(local_model)
                avg_loss.add(loss)
                all_per_accs.append(per_accs)

            train_loss = avg_loss.item()
            per_accs = list(np.array(all_per_accs).mean(axis=0))

            clients_end_time = time.time()
            clients_per_time = clients_end_time - clients_start_time
            client_total_time += clients_per_time

            time_diff(clients_start_time, clients_end_time)



            self.update_global(
                r=r,
                global_model=self.model,
                local_models=local_models,
                local_nums=local_trainData
            )

            # np.save("/data/yaominghao/code/FedRepo/result/SV_NonIID01" + self.args.filename + '.npy', self.history_sv)
            end_time = time.time()
            All_per_time = end_time - clients_start_time
            server_per_time = All_per_time - clients_per_time
            server_total_time += server_per_time

            if r % self.args.test_round == 0:
                # global test loader
                glo_test_acc, glo_test_precision, glo_test_recall, glo_test_f1 = self.test(
                    model=self.model,
                    loader=self.glo_test_loader,
                )

                # add to log
                self.logs["ROUNDS"].append(r)
                self.logs["LOSSES"].append(train_loss)
                self.logs["GLO_TACCS"].append(glo_test_acc)
                self.logs["GLO_precision"].append(glo_test_precision)
                self.logs["GLO_recall"].append(glo_test_recall)
                self.logs["GLO_f1"].append(glo_test_f1)

                np.save(
                    "/data/yaominghao/logs_2024/log202403/Contribution Martix/contribution_matrix_" + 'FedSV_' + self.args.filename + '_mydata' + '.npy',
                    self.history_sv)

                print("contribution_matrix save success!", self.history_sv)


                print('#################共训练了{}s#################'.format(end_time - start_time))

                print('#################本轮客户端共训练了{}s#################'.format(clients_per_time))
                print('#################本轮服务端共训练了{}s#################'.format(server_per_time))

                print('#################客户端共训练了{}s#################'.format(client_total_time))
                print('#################服务端共训练了{}s#################'.format(server_total_time))

                print("[R:{}] [Ls:{}] [TeAc:{}] [PAcBeg:{} PAcAft:{}] [pre: {}] [recall: {}] [f1_score: {}]".format(
                    r, train_loss, glo_test_acc, per_accs[0], per_accs[-1], glo_test_precision, glo_test_recall, glo_test_f1
                ))


    def update_local(self, r, model, train_loader, test_loader):
        # lr = min(r / 10.0, 1.0) * self.args.lr
        lr = self.args.lr

        optimizer = construct_optimizer(
            model, lr, self.args
        )

        if self.args.local_steps is not None:
            n_total_bs = self.args.local_steps
        elif self.args.local_epochs is not None:
            n_total_bs = max(
                int(self.args.local_epochs * len(train_loader)), 5
            )
        else:
            raise ValueError(
                "local_steps and local_epochs must not be None together"
            )

        model.train()

        loader_iter = iter(train_loader)

        avg_loss = Averager()
        per_accs = []

        for t in range(n_total_bs + 1):
            if t in [0, n_total_bs]:
                per_acc, precision, recall, f1 = self.test(
                    model=model,
                    loader=test_loader,
                )
                per_accs.append(per_acc)

            if t >= n_total_bs:
                break

            model.train()
            try:
                batch_x, batch_y = next(loader_iter)
            except Exception:
                loader_iter = iter(train_loader)
                batch_x, batch_y = next(loader_iter)

            if self.args.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            hs, logits = model(batch_x)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            optimizer.step()

            avg_loss.add(loss.item())

        loss = avg_loss.item()
        return model, per_accs, loss

    def update_global(self, r, global_model, local_models, local_nums):

        mean_state_dict = {}

        for name, param in global_model.state_dict().items():
            vs = []
            for client in local_models.keys():
                vs.append(local_models[client].state_dict()[name])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()
            mean_state_dict[name] = mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)

        sv = self.vaild(local_models=local_models,
                        global_model=copy.deepcopy(self.model),
                        fraction=local_nums,
                        r=r)
        print('Sharply Value', sv)

        all_sv = []
        for i in range(len(self.clients)):
            if i in sv:
                all_sv.append(sv[i])
            else:
                all_sv.append(0)
        self.history_sv.append(all_sv)

        round_end_time = time.time()
        round_run_time = round_end_time - self.start_time
        print('第{}轮运行时间为{}s'.format(r, round_run_time))

    def vaild(self, global_model, local_models, fraction, r):
        local_modelIdx = local_models.keys()
        arrange_modelIdx = list(local_modelIdx)[:5]
        random_modelIdx = list(local_modelIdx)[-5:]
        random_modeldictIdx = dict.fromkeys(random_modelIdx)
        dict_keys_format = random_modeldictIdx.keys()
        powerset = list(powersettool(dict_keys_format))              #生成所有组合

        # 要添加的元素
        additional_elements = tuple(arrange_modelIdx)
        # 使用列表推导式和元组拆包为每个元组添加元素
        modified_list = [(tup + additional_elements) for tup in powerset]


        # net_glob = net_glob.to(args.device)
        accuracy_dict = {}
        # start_time = time.time()  # start the timer

        # accuracy for the full dataset is the same as global accuracy
        alltest_acc, precision, recall, f1 = self.test(
                    model=global_model,
                    loader=self.glo_test_loader,)

        max_acc = alltest_acc
        max_set = []
        max_setModel = copy.deepcopy(global_model)

        # subset = powerset[1:-1]
        # sampled_elements = random.sample(subset, 30)
        # sampled_list = []
        # sampled_list.append(powerset[0])
        # for i in range(len(sampled_elements)):
        #     sampled_list.append(sampled_elements[i])
        # sampled_list.append(powerset[-1])

        # Federated Exact Algorithm
        for subset in modified_list[0:-1]:              #遍历所有组合计算ACC
            print("current subset: ", subset) # print check
            localSet_acc = 0
            local_weight = []
            sum = 0
            for idx in subset:
                sum += fraction[idx]
            for epoch in range(1):                #每个组合计算多次计算平均值
                local_sets, local_setAccs = [], []
                # print(f'\n | Global Training Round {subset} : {epoch+1} |\n') # print check

                # submodel_dict[subset].train()
                # note that the keys for train_dataset are [1,2,3,4,5]
                for idx in subset:
                    local_sets.append(copy.deepcopy(local_models[idx]))
                    local_weight.append(fraction[idx]/sum)

                # update global weights
                global_weights = self.aggregate(local_sets, local_weight)       #选出的局部聚合
                subacc, precision, recall, f1 = self.test(model=global_weights,            #测量组合的ACC
                                                loader=self.glo_test_loader,)
                local_setAccs.append(subacc)
                print(local_setAccs)

                # update global weights
                # submodel_dict[subset].load_state_dict(global_weights)
                if len(local_setAccs) == 1:
                    acc_avg = local_setAccs[0]
                else:
                    acc_avg = sum(local_setAccs) / len(local_setAccs)
                localSet_acc = acc_avg
                if max_acc < localSet_acc:
                    max_setModel = copy.deepcopy(global_weights)
                    max_acc = localSet_acc
                    max_set = subset

                    # Test inference after completion of training
            accuracy_dict[subset] = localSet_acc


        # accuracy for the random model
        accuracy_dict[modified_list[-1]] = alltest_acc      #计算全客户端的ACC

        self.model = max_setModel
        test_acc, precision, recall, f1 = self.test(model=self.model,
                    loader=self.glo_test_loader,)
        print('------------------[Rounds]:{}  [max acc]:{}------------------------'.format(r, test_acc))

        # random_acc,  precision, recall, f1 = self.test(model=self.initmodel,
        #                        loader=self.glo_test_loader,)
        # accuracy_dict[()] = random_acc


        contribution_value = []
        for value in accuracy_dict.values():
            contribution_value.append(value)

        contribution_dict = dict(zip(powerset, contribution_value))





        shapley_dict = shapley(contribution_dict, random_modelIdx, int((self.args.c_ratio * len(self.clients))/2))
        # print('shapley value', shapley_dict)
        return shapley_dict



    def aggregate(self, models, p):
        return fmodule._model_average(models, p)

    # def test(self, model, loader):
    #     model.eval()
    #
    #     acc_avg = Averager()
    #
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y) in enumerate(loader):
    #             if self.args.cuda:
    #                 batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
    #             _, logits = model(batch_x)
    #             acc = count_acc(logits, batch_y)
    #             acc_avg.add(acc)
    #
    #     acc = acc_avg.item()
    #     return acc

    def test(self, model, loader):
        model.eval()

        acc_avg = Averager()
        precision_avg = Averager()
        recall_avg = Averager()
        f1_avg = Averager()
        auc_avg = Averager()

        roc_auc = dict()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                _, logits = model(batch_x)
                acc = count_acc(logits, batch_y)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_pred = torch.argmax(logits, dim=1).to('cpu')
                    y_scores = torch.nn.functional.softmax(logits, dim=1).to('cpu')
                    y_scores = y_scores.numpy()
                    # print(y_scores.shape[1])
                    y_test = batch_y.to('cpu')
                    class_100 = np.arange(100)
                    class_10 = np.arange(10)
                    # y_test_one_hot = label_binarize(y_test, classes=np.unique(y_test))
                    y_test_one_hot = label_binarize(y_test, classes=class_10)
                    # print('y_test', np.unique(y_test))
                    # print(y_test_one_hot.shape[1])

                    precision = precision_score(y_test, y_pred, average='macro')

                    recall = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')

                # roc_auc = dict()
                #
                # # 计算每个类别的 ROC 曲线和 AUC
                # for j in range(self.args.n_classes):
                #     # print('j', j)
                #     y_tmp = y_test_one_hot[:, j]
                #     # print(y_tmp)
                #     y_score_tmp = y_scores[:, j]
                #     # print(y_score_tmp)
                #     auc_score = roc_auc_score(y_tmp, y_score_tmp)
                #     roc_auc[j] = auc_score
                #
                # mean_auc = np.mean(list(roc_auc.values()))

                acc_avg.add(acc)
                precision_avg.add(precision)
                recall_avg.add(recall)
                f1_avg.add(f1)
                # auc_avg.add(mean_auc)


        acc = acc_avg.item()
        precision = precision_avg.item()
        recall = recall_avg.item()
        f1 = f1_avg.item()
        # auc = auc_avg.item()


        # for i, auc_score in enumerate(roc_auc.values()):
        #     print(f'AUC for class {i}: {auc_score}')



        return acc, precision, recall, f1
        # return acc

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.args))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)

    def calculate_svalue(self, models):
        """
        Computes the Shapley Value for clients
        Parameters:
        models (dict): Key value pair of client identifiers and model updates.
        model_evaluation_func (func) : Function to evaluate model update.
        averaging_func (func) : Function to used to average the model updates.
        Returns:
        svalue: Key value pair of client identifiers and the computed shapley values.
        """

        # generate possible permutations
        all_perms = list(itertools.permutations(list(models.keys())))
        marginal_contributions = []

        # history map to avoid retesting the models
        history = {}

        for perm in all_perms:
            perm_values = {}
            local_models = {}

            for client_id in perm:
                model = copy.deepcopy(models[client_id])
                local_models[client_id] = model

                # get the current index eg: (A,B,C) on the 2nd iter, the index is (A,B)
                if len(perm_values.keys()) == 0:
                    index = (client_id,)
                else:
                    index = tuple(sorted(list(tuple(perm_values.keys()) + (client_id,))))

                if index in history.keys():
                    current_value = history[index]
                else:
                    model = fmodule._model_average(list(local_models.values()))
                    current_value = self.test(
                    model= model,
                    loader=self.glo_test_loader,
                )
                    history[index] = current_value

                perm_values[client_id] = max(0, current_value - sum(perm_values.values()))

            marginal_contributions.append(perm_values)

        svalue = {client_id: 0 for client_id in models.keys()}

        # sum the marginal contributions
        for perm in marginal_contributions:
            for key, value in perm.items():
                svalue[key] += value

        # compute the average marginal contribution
        svalue = {key: value / len(marginal_contributions) for key, value in svalue.items()}

        return svalue