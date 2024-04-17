import copy
import numpy as np
import random
import cvxopt
import fmodule
from cvxopt import matrix

import torch
import torch.nn as nn
from torch.linalg import norm
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs
from utils import get_dict_ID, get_dict_value

from tools import construct_dataloaders
from tools import construct_optimizer
from tools import flatten, unflatten
from tools import cal_client_contribution

from utils import time_diff
from datetime import datetime
import warnings
import time


class FedMDFG():
    def __init__(
        self, csets, gset, model, args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args

        self.clients = list(csets.keys())
        self.sam_clients = []

        self.theta = 11.25
        self.s = 5

        self.last_client_id_list = None
        self.last_g_locals = None
        self.last_d = None
        self.client_expected_loss = [None] * len(self.clients)
        self.client_join_count = [0] * len(self.clients)
        self.same_user_flag = True
        self.prefer_active = 0

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
        start_time = time.time()       #客户端开始时间
        print('#################开始训练时间为{}##################'.format(datetime.fromtimestamp(start_time)))
        ALLContribution_Matrix = []

        # Training
        for r in range(1, self.args.max_round + 1):
            n_sam_clients = int(self.args.c_ratio * len(self.clients))
            self.sam_clients = np.random.choice(
                self.clients, n_sam_clients, replace=False
            )

            local_models = {}
            local_loss = {}

            avg_loss = Averager()
            all_per_accs = []

            # 计算权重
            clients_weight = []

            clients_start_time = time.time()
            for client in self.sam_clients:
                local_model, per_accs, loss = self.update_local(
                    model=copy.deepcopy(self.model),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                )

                local_models[client] = copy.deepcopy(local_model)
                local_loss[client] = loss

                avg_loss.add(loss)
                all_per_accs.append(per_accs)

                cur_client_weight = len(self.csets[client][0])
                clients_weight.append(cur_client_weight)

            train_loss = avg_loss.item()
            per_accs = list(np.array(all_per_accs).mean(axis=0))

            clients_weight = [i / sum(clients_weight) for i in clients_weight]

            #计算贡献值
            Allclient_contribution = cal_client_contribution(self.model, self.args.n_clients, local_models)
            ALLContribution_Matrix.append(Allclient_contribution)

            clients_end_time = time.time()    #客户端结束时间
            clients_per_time = clients_end_time - clients_start_time
            client_total_time += clients_per_time

            time_diff(clients_start_time, clients_end_time)

            self.update_global(
                r=r,
                global_model=self.model,
                local_models=local_models,
                local_loss=local_loss,
                clients_weight=clients_weight
            )

            end_time = time.time()      #服务端结束时间

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

                # np.save("/data/yaominghao/code/FedRepo/result/contribution_matrix_Allmydatarefedfv_" + self.args.filename + '.npy', ALLContribution_Matrix)
                np.save(
                    "/data/yaominghao/logs_2024/logs_202403/Contribution Martix/contribution_matrix_" + 'FedMDFG_' + self.args.filename + '_mydataTFCNN' + '.npy',
                    ALLContribution_Matrix)
                print("contribution_matrix save success!")

                All_per_time = end_time - clients_start_time
                server_per_time = All_per_time - clients_per_time
                server_total_time += server_per_time


                print('#################共训练了{}s#################'.format(end_time - start_time))

                print('#################本轮客户端共训练了{}s#################'.format(clients_per_time))
                print('#################本轮服务端共训练了{}s#################'.format(server_per_time))

                print('#################客户端共训练了{}s#################'.format(client_total_time))
                print('#################服务端共训练了{}s#################'.format(server_total_time))

                print("[R:{}] [Ls:{}] [TeAc:{}] [PAcBeg:{} PAcAft:{}] [pre: {}] [recall: {}] [f1_score: {}]".format(
                    r, train_loss, glo_test_acc, per_accs[0], per_accs[-1], glo_test_precision, glo_test_recall, glo_test_f1
                ))

    def update_local(self, model, train_loader, test_loader):
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

        weights = []
        grad_mat = []

        for t in range(n_total_bs + 1):
            if t in [0, n_total_bs]:
                per_acc = self.test(
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

            weights.append(batch_y.shape[0])

            hs, logits = model(batch_x)

            criterion = nn.CrossEntropyLoss()

            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            grad_vec = flatten(model)
            grad_vec_norm = torch.norm(grad_vec)
            grad_mat.append(grad_vec)
            optimizer.step()

            avg_loss.add(loss.item())

        loss = avg_loss.item()
        print('femdfg loss', loss)


        grad_mat = torch.stack([grad_mat[i] for i in range(len(grad_mat))])
        weights = torch.Tensor(weights).float().to(grad_mat.device)
        weights = weights / torch.sum(weights)
        g = weights @ grad_mat
        model_norm = torch.norm(g)
        return model, per_accs, loss

    def update_global(self, r, global_model, local_models, local_loss, clients_weight):
        g_locals, l_locals = get_dict_value(local_models), torch.tensor(get_dict_value(local_loss))
        g_locals = torch.stack([flatten(i) for i in g_locals])
        # clients_weight = torch.tensor(clients_weight).to('cuda:0')
        # g_locals = g_locals_tmp * clients_weight.view(-1, 1)

        client_id_list = get_dict_ID(local_models)
        force_active = False
        increase_count = 0
        for i, client_id in enumerate(client_id_list):
            if self.client_join_count[client_id] == 0:
                self.client_expected_loss[client_id] = l_locals[i]
            else:
                if l_locals[i] <= self.client_expected_loss[client_id]:
                    self.client_expected_loss[client_id] = (self.client_expected_loss[client_id] * self.client_join_count[client_id] + l_locals[i]) / (self.client_join_count[client_id] + 1)
                else:
                    if l_locals[i] > self.client_expected_loss[client_id]:
                        increase_count += 1
            self.client_join_count[client_id] += 1
        if increase_count > 0 and increase_count < len(client_id_list):
            force_active = True
        # historical fairness
        if self.last_client_id_list is not None:
            add_idx = []
            for idx, last_client_id in enumerate(self.last_client_id_list):
                if last_client_id not in client_id_list:
                    add_idx.append(idx)
            if len(add_idx) > 0:
                add_grads = self.last_g_locals[add_idx, :]
                self.same_user_flag = False
            else:
                add_grads = None
                self.same_user_flag = True
        else:
            add_grads = None
            self.same_user_flag = True
        # grad_local_norm = torch.norm(g_locals, dim=1)
        grad_local_norm = torch.tensor([v.norm().item() for v in g_locals])
        live_idx = torch.where(grad_local_norm > 1e-6)[0]
        if len(live_idx) == 0:
            return
        if len(live_idx) > 0:
            g_locals = g_locals[live_idx, :]
            l_locals = l_locals[live_idx]
            grad_local_norm = torch.norm(g_locals, dim=1)
        # scale the outliers of all gradient norms
        miu = torch.mean(grad_local_norm)
        print('miu', miu)
        g_locals = g_locals / grad_local_norm.reshape(-1, 1) * miu

        fair_guidance_vec = torch.Tensor([1.0] * len(live_idx))
        # calculate d
        d, vec, p_active_flag, fair_grad = self.get_fedmdfg_d(g_locals, l_locals, add_grads, self.theta, fair_guidance_vec, force_active)
        if p_active_flag == 1:
            self.prefer_active = 1
        # Update parameters of the model
        # line search
        weights = torch.Tensor([1 / len(live_idx)] * len(live_idx)).float().to('cuda:0')

        g_norm = torch.norm(weights @ g_locals)
        g_norm_tmp = torch.norm(g_locals * weights.view(-1, 1))
        d_norm = torch.norm(d)
        min_lr = self.args.lr
        d_old = copy.deepcopy(d)
        d = d / d_norm * g_norm
        # prevent the effects of the float or double type, it can be sikpped theoretically
        while torch.max(-(vec @ d)) > 1e-6:
            if torch.norm(d) > d_norm * 2:
                d /= 2
            else:
                d = d_old
                break
        scale = torch.norm(d) / torch.norm(d_old)
        self.line_search(g_locals, d, fair_guidance_vec, fair_grad, min_lr, l_locals, live_idx, scale)
        # self.current_training_num += 1
        self.last_client_id_list = get_dict_ID(local_models)
        self.last_client_id_list = [self.last_client_id_list[live_idx[i]] for i in range(len(live_idx))]
        self.last_g_locals = copy.deepcopy(g_locals)
        self.last_d = d

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

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.args))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)

    def get_fedmdfg_d(self, grads, value, add_grads, alpha, fair_guidance_vec, force_active):
        fair_grad = None
        value_norm = torch.norm(value)
        norm_values = value / value_norm
        fair_guidance_vec /= torch.norm(fair_guidance_vec)
        cos = float(norm_values @ fair_guidance_vec)
        cos = min(1, cos)
        cos = max(-1, cos)
        bias = np.arccos(cos) / np.pi * 180
        pref_active_flag = (bias > alpha) | force_active
        norm_vec = torch.norm(grads, dim=1)
        indices = list(range(len(norm_vec)))
        grads = norm_vec[indices].reshape(-1, 1) * grads / (norm_vec + 1e-6).reshape(-1, 1)
        if not pref_active_flag:
            vec = grads
            pref_active_flag = 0
        else:
            pref_active_flag = 1
            h_vec = (fair_guidance_vec @ norm_values * norm_values - fair_guidance_vec).reshape(1, -1)
            h_vec /= torch.norm(h_vec)
            h_vec = h_vec.to('cuda:0')
            fair_grad = h_vec @ grads
            vec = torch.cat((grads, fair_grad))
        if add_grads is not None:
            norm_vec = torch.norm(add_grads, dim=1)
            indices = list(range(len(norm_vec)))
            random.shuffle(indices)
            add_grads = norm_vec[indices].reshape(-1, 1) * add_grads / (norm_vec + 1e-6).reshape(-1, 1)
            vec = torch.vstack([vec, add_grads])
        sol, _ = self.setup_qp_and_solve(vec.cpu().detach().numpy())
        sol = torch.from_numpy(sol).float().to('cuda:0')
        d = sol @ vec

        return d, vec, pref_active_flag, fair_grad

    def setup_qp_and_solve(self, vec):
        P = np.dot(vec, vec.T)
        n = P.shape[0]
        q = np.zeros(n)
        G = - np.eye(n)
        h = np.zeros(n)
        A = np.ones((1, n))
        b = np.ones(1)
        cvxopt.solvers.options['show_progress'] = False
        sol, optimal_flag = self.cvxopt_solve_qp(P, q, G, h, A, b)
        return sol, optimal_flag

    def cvxopt_solve_qp(self, P, q, G=None, h=None, A=None, b=None):
        P = 0.5 * (P + P.T)
        P = P.astype(np.double)
        q = q.astype(np.double)
        args = [matrix(P), matrix(q)]
        if G is not None:
            args.extend([matrix(G), matrix(h)])
            if A is not None:
                args.extend([matrix(A), matrix(b)])
        sol = cvxopt.solvers.qp(*args)
        optimal_flag = 1
        if 'optimal' not in sol['status']:
            optimal_flag = 0
        return np.array(sol['x']).reshape((P.shape[1],)), optimal_flag

    def line_search(self, g_locals, d, fair_guidance_vec, fair_grad, base_lr, l_locals_0, live_idx, scale):
        old_loss_norm = float(torch.sum(l_locals_0))
        fair_guidance_vec_norm = torch.norm(fair_guidance_vec)
        old_cos = l_locals_0 @ fair_guidance_vec / (old_loss_norm * fair_guidance_vec_norm)
        beta = 1e-4
        c = -(g_locals@d)
        print('self.same_user_flag:',self.same_user_flag)
        if self.same_user_flag:
            lr = float(2**self.s * base_lr)
        else:
            lr = float(base_lr)
        old_model = copy.deepcopy(self.model.state_dict())
        min_lr = float(0.5**self.s * base_lr / scale)
        lr_storage = []
        norm_storage = []
        while lr >= min_lr:
            print('lr', lr, 'min_lr', min_lr)
            self.model.load_state_dict(old_model)
            temp_model = self.update_model(self.model, d, lr)
            # evaluate temporary model
            # Note that here we use such a way just for convenient, that we reuse the framework to copy the model to all clients.
            # In fact, we don't need this step, just send the direction d^t to clients before the step size line search,
            # and then just send lr to clients and let clients update a local temporary model by theirselves instead.
            # self.send_sync_model(update_count=False, model=temp_model)
            # self.send_cal_all_batches_loss_order()
            # l_locals = self.send_require_cal_all_batches_loss_result()
            cur_local_models = {}
            cur_local_loss = {}
            print('evaluate ')
            for client in self.sam_clients:
                local_model, per_accs, loss = self.update_local(
                    model=copy.deepcopy(temp_model),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                )

                cur_local_models[client] = copy.deepcopy(local_model)
                cur_local_loss[client] = loss

            l_locals = torch.tensor(get_dict_value(cur_local_loss))
            l_locals = l_locals[live_idx]
            # store the loss norm
            l_locals_norm = float(torch.sum(l_locals))
            lr_storage.append(lr)
            norm_storage.append(l_locals_norm)
            # stop criterion
            param = lr * beta * c
            print('prefer_active: ', self.prefer_active)
            print('l_locals_0 - l_locals:  ', l_locals_0 - l_locals)
            print('lr * beta * c: ', lr * beta * c,)
            print('l_locals_0 - l_locals >= lr * beta * c ',  l_locals_0 - l_locals >= param.to('cpu'))
            print('param: ', (l_locals @ fair_guidance_vec) / (torch.norm(l_locals) * fair_guidance_vec_norm) - old_cos)
            if self.prefer_active == 0 and torch.all(l_locals_0 - l_locals >= param.to('cpu')):
                lr_storage = []
                norm_storage = []
                break
            elif self.prefer_active == 1 and torch.all(l_locals_0 - l_locals >= param.to('cpu')) and (l_locals @ fair_guidance_vec) / (torch.norm(l_locals) * fair_guidance_vec_norm) - old_cos > 0:
                lr_storage = []
                norm_storage = []
                break
            lr /= 2
        if len(norm_storage) > 0:
            for idx, l_locals_norm in enumerate(norm_storage):
                lr = lr_storage[idx]
                if lr > base_lr and self.same_user_flag == False:
                    continue
                if l_locals_norm < old_loss_norm:
                    norm_storage = []
                    break
        if len(norm_storage) > 0:
            best_idx = np.argmin(norm_storage)
            lr = lr_storage[best_idx]
        self.model.load_state_dict(old_model)
        print('aggregation ')
        self.model = self.update_model(self.model, d, lr)
        # self.model = self.update_model(self.model, d, self.args.lr)

    def update_model(self, model, d, lr):
        optimizer = construct_optimizer(
            model, lr, self.args
        )
        # for i, p in enumerate(model.parameters()):
        #     p.grad = d[self.model.Loc_reshape_list[i]]

        model = unflatten(d, copy.deepcopy(model))

        param_state_dict = {}
        for name, param in model.state_dict().items():
            n_params = len(param.view(-1))
            value = torch.as_tensor(d[:n_params]).reshape(param.size())
            param_state_dict[name] = value
            d = d[n_params:]

        model.load_state_dict(param_state_dict, strict=False)

        optimizer.step()

        return model


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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
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




