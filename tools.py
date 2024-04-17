import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
import math

import matplotlib.pyplot as plt


def guassian_kernel(
        source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    L2_distance = ((
        total.unsqueeze(dim=1) - total.unsqueeze(dim=0)
    ) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth += 1e-8

    # print("Bandwidth:", bandwidth)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-L2_distance / band) for band in bandwidth_list
    ]
    return sum(kernel_val)


def mmd_rbf_noaccelerate(
        source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(
        source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num,
        fix_sigma=fix_sigma
    )
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def construct_dataloaders(clients, csets, gset, args):
    train_loaders = {}
    test_loaders = {}
    glo_test_loader = None

    for client in clients:
        assert isinstance(csets[client], tuple), \
            "csets must be a tuple (train_set, test_set): {}".format(client)

        assert csets[client][1] is not None, \
            "local test set must not be None in client: {}".format(client)

        train_loader = DataLoader(
            csets[client][0],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        train_loaders[client] = train_loader

        test_loader = DataLoader(
            csets[client][1],
            batch_size=args.batch_size * 10,
            shuffle=False,
            drop_last=False,
        )
        test_loaders[client] = test_loader

    assert gset is not None, \
        "global test set must not be None"

    glo_test_loader = DataLoader(
        gset,
        batch_size=args.batch_size * 10,
        shuffle=False,
        drop_last=False,
    )

    return train_loaders, test_loaders, glo_test_loader


def construct_optimizer(model, lr, args):
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError("No such optimizer: {}".format(
            args.optimizer
        ))
    return optimizer


#计算客户端和全局原型相似度(前轮平均原型改进)
def client_similarity_P(preclient_features, clients_features, clients_class_num, classes, clients, r):
    """
    input:
        preclient_features: 上一轮的原型，轮次为1尺寸为 [clients_num, class_num, prototype_size], 其他为[class_num, prototype_size]
        clients_features: 当前轮次各个客户端的原型，尺寸为[clients_num, class_num, prototype_size]
    output:
        loss值
    """
    # 计算前一轮平均类原型  output_size=[class_num, prototype_size]
    if r == 1:     #第一轮初始化
        pre_classes_avg = [None for j in range(classes)]
        # 求每个类型的avg_feature
        for feature in range(classes):
            cnt = 0
            for client in range(clients):
                if preclient_features[client] is None or len(preclient_features[client]) == 0:       #判断该客户端是否被选中
                    continue
                client_feature = preclient_features[client][feature]  #记录当前客户端的当前类
                if client_feature is None:                #判断该被选中的客户端是否存在当前类
                    continue
                if pre_classes_avg[feature] is None:      #若是第一个添加的直接加入
                    pre_classes_avg[feature] = client_feature
                    cnt += 1
                elif pre_classes_avg[feature].shape == client_feature.shape:
                    pre_classes_avg[feature] = torch.add(pre_classes_avg[feature], client_feature)   #将当前特征加入
                    cnt += 1
            if pre_classes_avg[feature] is not None:
                pre_classes_avg[feature] = torch.div(pre_classes_avg[feature], cnt)   #计算出来的平均特征
    else:
        pre_classes_avg = preclient_features           #初始化后面的轮次直接使用上一轮的类平均原型

    #计算当前客户端类原型和前一轮平均类原型之间的差值的二范数（p(t-1)_avg - pt）  output_size=[class_num, clients_num]
    allClass_change = [[] for j in range(classes)]
    allClass_changeSum = [[] for j in range(classes)]
    for feature in range(classes):
        allClass_changeSum[feature] = 0
        curClass_change = [0 for i in range(clients)]
        for client in range(clients):
            if pre_classes_avg[feature] is None:         #判断上一轮是否有该类的平均原型
                if len(clients_features[client]) == 0:   #判断当前轮次的该客户端是否被选中，若未被选中变化为0
                    curClass_change[client] = 0
                else:
                    if clients_features[client][feature] is None:    #若选中了没有当前类，变化为0
                        curClass_change[client] = 0
                    else:
                        curClass_change[client] = torch.norm(clients_features[client][feature])
                        allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
            else:
                if len(clients_features[client]) == 0:
                    # curClass_change[client] = torch.norm(pre_classes_avg[feature])
                    # allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
                    curClass_change[client] = 0
                else:
                    if clients_features[client][feature] is None:
                        curClass_change[client] = torch.norm(pre_classes_avg[feature])
                        allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
                    else:
                        curClass_change[client] = torch.norm(
                            torch.sub(pre_classes_avg[feature], clients_features[client][feature]))
                        allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
            allClass_change[feature] = curClass_change

    #归一化change    output_size=Norm([class_num, clients_num])
    allClass_changeNorm = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_change = [0 for i in range(clients)]
        curClass_sum = sum(allClass_change[feature])      #计算当前类的原型距离之和
        for client in range(clients):
            if allClass_change[feature][client] is None or allClass_change[feature][client] == 0:
                continue
            elif allClass_change[feature][client] is not None:
                curClass_change_tmp = allClass_change[feature][client]
                curClass_change_norm = curClass_change_tmp/curClass_sum   #归一化
                curClass_change[client] = curClass_change_norm
        allClass_changeNorm[feature] = curClass_change

    # 求每个类型的avg_feature    output_size=[class_num, prototype_size]
    classes_avg = [None for j in range(classes)]
    for feature in range(classes):
        cnt = 0
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue

            client_feature = clients_features[client][feature]  #记录当前客户端的当前类

            if client_feature is None:
                continue
            if classes_avg[feature] is None:
                classes_avg[feature] = client_feature
                cnt += 1
            elif classes_avg[feature].shape == client_feature.shape:
                classes_avg[feature] = torch.add(classes_avg[feature], client_feature)   #将当前特征加入
                cnt += 1
        if classes_avg[feature] is not None:
            classes_avg[feature] = torch.div(classes_avg[feature], cnt)   #计算出来的平均特征

    #求每个客户端的每个类到全局该类平均特征的距离（余弦）   output_size=[class_num, clients_num]
    allClass_distance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_distance = [0 for i in range(clients)]
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue
            if clients_features[client][feature] is None:
                continue
            if classes_avg[feature] is not None and classes_avg[feature].shape == clients_features[client][feature].shape:
                cosine_similarity = torch.cosine_similarity(clients_features[client][feature].unsqueeze(0).float(), classes_avg[feature].unsqueeze(0).float())
                # cosine_similarity = torch.mean(cosin_smilarity)
                # if cosine_similarity.item() > 0:                  #对余弦距离做筛选
                #     curClass_distance[client] = cosine_similarity.item()
                curClass_distance[client] = cosine_similarity.item()
        allClass_distance[feature] = curClass_distance
        if min(allClass_distance[feature]) < 0:                       ##对余弦距离做筛选(保留负值数轴往右+min)
            allClass_distance[feature] + abs(min(allClass_distance[feature]))

    #求全局归一化计算后的余弦距离     output_size=Norm([class_num, clients_num])
    allClass_consine_norm = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_cmpDistance = [0 for i in range(clients)]
        curClass_sum = sum(allClass_distance[feature])      #计算当前类的原型距离之和
        for client in range(clients):
            if allClass_distance[feature][client] is None or allClass_distance[feature][client] == 0:
                continue
            elif allClass_distance[feature][client] is not None:
                curcosine_similarity = allClass_distance[feature][client]
                cosine_cmpSimilarity = curcosine_similarity/curClass_sum   #归一化
                curClass_cmpDistance[client] = cosine_cmpSimilarity
        allClass_consine_norm[feature] = curClass_cmpDistance

    #归一化后的各个客户端的余弦距离与当前轮次客户端原型相乘得到计算后的原型距离     output_size=[class_num, clients_num, prototype_size]
    allClass_cmpDistance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_cmpDistance = [0 for i in range(clients)]
        for client in range(clients):
            if allClass_distance[feature][client] is None or allClass_distance[feature][client] == 0:
                continue
            elif allClass_distance[feature][client] is not None:
                curcosine_norm = allClass_consine_norm[feature][client]
                client_features_tmp = clients_features[client][feature]
                cosine_cmpSimilarity = curcosine_norm * client_features_tmp   #consine_norm * prototype
                curClass_cmpDistance[client] = cosine_cmpSimilarity
        allClass_cmpDistance[feature] = curClass_cmpDistance

    # 求计算后每个类型的avg_feature     output_size=[class_num, prototype_size]
    classes_cmpAvg = [None for j in range(classes)]
    for feature in range(classes):
        cnt = 0
        for client in range(clients):
            if allClass_cmpDistance[feature][client] is None:
                continue

            cur_feature_client = allClass_cmpDistance[feature][client]  #记录当前客户端的当前类

            if cur_feature_client is None:
                continue

            if type(cur_feature_client) == int:
                continue

            if classes_cmpAvg[feature] is None:
                classes_cmpAvg[feature] = cur_feature_client
                cnt += 1
            elif classes_cmpAvg[feature].shape == cur_feature_client.shape:
                classes_cmpAvg[feature] = torch.add(classes_cmpAvg[feature], cur_feature_client)   #将当前特征加入
                cnt += 1
        # if classes_cmpAvg[feature] is not None:
        #     classes_cmpAvg[feature] = torch.div(classes_cmpAvg[feature], cnt)   #计算出来的平均特征

    #计算客户端类原型到计算后的原型的余弦距离     output_size=[class_num, clients_num]
    allClass_tocmpDistance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_tocmpDistance = [0 for i in range(clients)]
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue
            if clients_features[client][feature] is None:
                continue
            if classes_cmpAvg[feature] is not None and classes_cmpAvg[feature].shape == clients_features[client][feature].shape:
                cosine_similarity = torch.mean(torch.cosine_similarity(clients_features[client][feature].unsqueeze(0).float(), classes_cmpAvg[feature].unsqueeze(0).float()))
                # if cosine_similarity.item() > 0:
                #     curClass_tocmpDistance[client] = cosine_similarity.item()
                curClass_tocmpDistance[client] = cosine_similarity.item()
        allClass_tocmpDistance[feature] = curClass_tocmpDistance
        if min(allClass_tocmpDistance[feature]) < 0:                       ##对余弦距离做筛选(保留负值数轴往右+min)
            allClass_tocmpDistance[feature] + abs(min(allClass_tocmpDistance[feature]))

    #归一化第二次的距离作为相似度（贡献）   output_size=Norm([class_num, clients_num])
    allClass_normSimilarities = [[] for j in range(classes)]
    for feature in range(classes):
        class_normSimilarities = allClass_tocmpDistance[feature]
        # for client in range(clients):
        #     nums = clients_class_num[client]
        #     if len(nums) == 0:
        #         class_normSimilarities[client] = 0
        #     else:
        #         # class_normSimilarities[client] = class_normSimilarities[client] * nums[feature]      #查看一下公式是否需要num[feature]
        #         class_normSimilarities[client] = class_normSimilarities[client]      #查看一下公式是否需要num[feature]
        class_sum = sum(class_normSimilarities)
        for client in range(clients):
            if class_sum == 0:
                class_normSimilarities[client] = 0
            else:
                class_normSimilarities[client] = class_normSimilarities[client]/class_sum
        allClass_normSimilarities[feature] = class_normSimilarities

    #原型差值*相似度（贡献）   output_size=[class_num, clients_num]
    allClass_normChange = [[] for j in range(classes)]
    allClass_normchangeSum = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_normChange = [0 for i in range(clients)]
        allClass_normchangeSum[feature] = 0
        for client in range(clients):
            if allClass_change[feature][client] is None or allClass_change[feature][client] == 0:
                continue
            elif allClass_change[feature][client] is not None:
                curfeature_client_normchange_tmp = allClass_change[feature][client]
                allClass_normSimilarities_tmp = allClass_normSimilarities[feature][client]
                curfeature_client_normchange = curfeature_client_normchange_tmp * allClass_normSimilarities_tmp # 二范数*贡献（范数再归一化一次）
                curClass_normChange[client] = curfeature_client_normchange.item()
                allClass_normchangeSum[feature] = torch.add(allClass_normchangeSum[feature], curClass_normChange[client])

        allClass_normChange[feature] = curClass_normChange

    #归一化最终乘积值     output_size=Norm([class_num, clients_num])
    allClass_final = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_final = [0 for i in range(clients)]
        for client in range(clients):
            if allClass_normChange[feature][client] is None or allClass_normChange[feature][client] == 0:
                continue
            elif allClass_normChange[feature][client] is not None:
                curfeature_client_normchange = allClass_normChange[feature][client]
                if (allClass_normChange[feature][client] != 0):
                    curfeature_client_normchange = curfeature_client_normchange / allClass_normchangeSum[feature] # 归一化
                    curClass_final[client] = curfeature_client_normchange.item()
                else:
                    continue
        allClass_final[feature] = curClass_final

    return allClass_final

#计算客户端和全局原型相似度(只有change)
def client_similarity_V(preclient_features, clients_features, clients_class_num, classes, clients, r):
    # 计算前一轮平均类原型  output_size=[class_num, prototype_size]
    if r == 1:     #第一轮初始化
        pre_classes_avg = [None for j in range(classes)]
        # 求每个类型的avg_feature
        for feature in range(classes):
            cnt = 0
            for client in range(clients):
                if preclient_features[client] is None or len(preclient_features[client]) == 0:       #判断该客户端是否被选中
                    continue
                client_feature = preclient_features[client][feature]  #记录当前客户端的当前类
                if client_feature is None:                #判断该被选中的客户端是否存在当前类
                    continue
                if pre_classes_avg[feature] is None:      #若是第一个添加的直接加入
                    pre_classes_avg[feature] = client_feature
                    cnt += 1
                elif pre_classes_avg[feature].shape == client_feature.shape:
                    pre_classes_avg[feature] = torch.add(pre_classes_avg[feature], client_feature)   #将当前特征加入
                    cnt += 1
            if pre_classes_avg[feature] is not None:
                pre_classes_avg[feature] = torch.div(pre_classes_avg[feature], cnt)   #计算出来的平均特征
    else:
        pre_classes_avg = preclient_features           #初始化后面的轮次直接使用上一轮的类平均原型

    #计算当前客户端类原型和前一轮平均类原型之间的差值的二范数（p(t-1)_avg - pt）  output_size=[class_num, clients_num]
    allClass_change = [[] for j in range(classes)]
    allClass_changeSum = [[] for j in range(classes)]
    for feature in range(classes):
        allClass_changeSum[feature] = 0
        curClass_change = [0 for i in range(clients)]
        for client in range(clients):
            if pre_classes_avg[feature] is None:         #判断上一轮是否有该类的平均原型
                if len(clients_features[client]) == 0:   #判断当前轮次的该客户端是否被选中，若未被选中变化为0
                    curClass_change[client] = 0
                else:
                    if clients_features[client][feature] is None:    #若选中了没有当前类，变化为0
                        curClass_change[client] = 0
                    else:
                        curClass_change[client] = torch.norm(clients_features[client][feature])
                        allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
            else:
                if len(clients_features[client]) == 0:
                    # curClass_change[client] = torch.norm(pre_classes_avg[feature])
                    # allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
                    curClass_change[client] = 0
                else:
                    if clients_features[client][feature] is None:
                        curClass_change[client] = torch.norm(pre_classes_avg[feature])
                        allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
                    else:
                        curClass_change[client] = torch.norm(
                            torch.sub(pre_classes_avg[feature], clients_features[client][feature]))
                        allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
            allClass_change[feature] = curClass_change

    # #归一化change
    # allClass_changeNorm = [[] for j in range(classes)]
    # for feature in range(classes):
    #     curClass_change = [0 for i in range(clients)]
    #     for client in range(clients):
    #         if allClass_change[feature][client] is None or allClass_change[feature][client] == 0:
    #             continue
    #         elif allClass_change[feature][client] is not None:
    #             curClass_change_tmp = allClass_change[feature][client]
    #             curClass_change_norm = curClass_change_tmp/allClass_changeSum[feature]   #归一化
    #             curClass_change[client] = curClass_change_norm.item()
    #     allClass_changeNorm[feature] = curClass_change

    # # 求原型之间的差值的二范数(老方法)
    # allClass_change = [[] for j in range(classes)]
    # allClass_changeSum = [[] for j in range(classes)]
    # for feature in range(classes):
    #     allClass_changeSum[feature] = 0
    #     curClass_change = [0 for i in range(clients)]
    #     for client in range(clients):
    #         if len(preclient_features[client]) == 0 and len(clients_features[client]) == 0:
    #             curClass_change[client] = 0
    #         elif len(preclient_features[client]) == 0:
    #             curClass_change[client] = torch.norm(clients_features[client] * (-1))
    #             allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
    #         elif len(clients_features[client]) == 0:
    #             curClass_change[client] = torch.norm(preclient_features[client])
    #             allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
    #         elif len(preclient_features[client]) != 0 and len(clients_features[client]) != 0:
    #             if preclient_features[client][feature] is None and clients_features[client][feature] is None:
    #                 curClass_change[client] = 0
    #             elif preclient_features[client][feature] is None:
    #                 curClass_change[client] = torch.norm(clients_features[client][feature])
    #                 allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
    #             elif clients_features[client][feature] is None:
    #                 curClass_change[client] = torch.norm(preclient_features[client][feature])
    #                 allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
    #             elif len(preclient_features[client][feature]) != 0 and len(clients_features[client][feature]) != 0:
    #                 curClass_change[client] = torch.norm(torch.sub(preclient_features[client][feature], clients_features[client][feature]))
    #                 allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
    #             else:
    #                 continue
    #         else:
    #             continue
    #         allClass_change[feature] = curClass_change

    #归一化最终乘积值
    allClass_final = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_final = [0 for i in range(clients)]
        for client in range(clients):
            if allClass_change[feature][client] is None or allClass_change[feature][client] == 0:
                continue
            elif allClass_change[feature][client] is not None:
                curfeature_client_normchange = allClass_change[feature][client]
                if (allClass_change[feature][client] != 0):
                    curfeature_client_normchange = curfeature_client_normchange / allClass_changeSum[feature] # 归一化
                    curClass_final[client] = curfeature_client_normchange.item()
                else:
                    continue
        allClass_final[feature] = curClass_final

    # return allClass_normSimilarities
    return allClass_final

#计算客户端和全局原型相似度(前轮平均原型改进后只要差值)
def client_similarity_M(preclient_features, clients_features, clients_class_num, classes, clients, r):
    # 求每个类型的avg_feature    output_size=[class_num, prototype_size]
    classes_avg = [None for j in range(classes)]
    for feature in range(classes):
        cnt = 0
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue

            client_feature = clients_features[client][feature]  #记录当前客户端的当前类

            if client_feature is None:
                continue
            if classes_avg[feature] is None:
                classes_avg[feature] = client_feature
                cnt += 1
            elif classes_avg[feature].shape == client_feature.shape:
                classes_avg[feature] = torch.add(classes_avg[feature], client_feature)   #将当前特征加入
                cnt += 1
        if classes_avg[feature] is not None:
            classes_avg[feature] = torch.div(classes_avg[feature], cnt)   #计算出来的平均特征

    #求每个客户端的每个类到全局该类平均特征的距离（余弦）   output_size=[class_num, clients_num]
    allClass_distance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_distance = [0 for i in range(clients)]
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue
            if clients_features[client][feature] is None:
                continue
            if classes_avg[feature] is not None and classes_avg[feature].shape == clients_features[client][feature].shape:
                cosine_similarity = torch.cosine_similarity(clients_features[client][feature].unsqueeze(0).float(), classes_avg[feature].unsqueeze(0).float())
                # cosine_similarity = torch.mean(cosin_smilarity)
                # if cosine_similarity.item() > 0:                  #对余弦距离做筛选
                #     curClass_distance[client] = cosine_similarity.item()
                curClass_distance[client] = cosine_similarity.item()
        allClass_distance[feature] = curClass_distance
        if min(allClass_distance[feature]) < 0:                       ##对余弦距离做筛选(保留负值数轴往右+min)
            allClass_distance[feature] + abs(min(allClass_distance[feature]))

    #求全局归一化计算后的余弦距离     output_size=Norm([class_num, clients_num])
    allClass_consine_norm = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_cmpDistance = [0 for i in range(clients)]
        curClass_sum = sum(allClass_distance[feature])      #计算当前类的原型距离之和
        for client in range(clients):
            if allClass_distance[feature][client] is None or allClass_distance[feature][client] == 0:
                continue
            elif allClass_distance[feature][client] is not None:
                curcosine_similarity = allClass_distance[feature][client]
                cosine_cmpSimilarity = curcosine_similarity/curClass_sum   #归一化
                curClass_cmpDistance[client] = cosine_cmpSimilarity
        allClass_consine_norm[feature] = curClass_cmpDistance

    #归一化后的各个客户端的余弦距离与当前轮次客户端原型相乘得到计算后的原型距离     output_size=[class_num, clients_num, prototype_size]
    allClass_cmpDistance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_cmpDistance = [0 for i in range(clients)]
        for client in range(clients):
            if allClass_distance[feature][client] is None or allClass_distance[feature][client] == 0:
                continue
            elif allClass_distance[feature][client] is not None:
                curcosine_norm = allClass_consine_norm[feature][client]
                client_features_tmp = clients_features[client][feature]
                cosine_cmpSimilarity = curcosine_norm * client_features_tmp   #consine_norm * prototype
                curClass_cmpDistance[client] = cosine_cmpSimilarity
        allClass_cmpDistance[feature] = curClass_cmpDistance

    # 求计算后每个类型的avg_feature     output_size=[class_num, prototype_size]
    classes_cmpAvg = [None for j in range(classes)]
    for feature in range(classes):
        cnt = 0
        for client in range(clients):
            if allClass_cmpDistance[feature][client] is None:
                continue

            cur_feature_client = allClass_cmpDistance[feature][client]  #记录当前客户端的当前类

            if cur_feature_client is None:
                continue

            if type(cur_feature_client) == int:
                continue

            if classes_cmpAvg[feature] is None:
                classes_cmpAvg[feature] = cur_feature_client
                cnt += 1
            elif classes_cmpAvg[feature].shape == cur_feature_client.shape:
                classes_cmpAvg[feature] = torch.add(classes_cmpAvg[feature], cur_feature_client)   #将当前特征加入
                cnt += 1
        # if classes_cmpAvg[feature] is not None:
        #     classes_cmpAvg[feature] = torch.div(classes_cmpAvg[feature], cnt)   #计算出来的平均特征

    #计算客户端类原型到计算后的原型的余弦距离     output_size=[class_num, clients_num]
    allClass_tocmpDistance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_tocmpDistance = [0 for i in range(clients)]
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue
            if clients_features[client][feature] is None:
                continue
            if classes_cmpAvg[feature] is not None and classes_cmpAvg[feature].shape == clients_features[client][feature].shape:
                cosine_similarity = torch.mean(torch.cosine_similarity(clients_features[client][feature].unsqueeze(0).float(), classes_cmpAvg[feature].unsqueeze(0).float()))
                # if cosine_similarity.item() > 0:
                #     curClass_tocmpDistance[client] = cosine_similarity.item()
                curClass_tocmpDistance[client] = cosine_similarity.item()
        allClass_tocmpDistance[feature] = curClass_tocmpDistance
        if min(allClass_tocmpDistance[feature]) < 0:                       ##对余弦距离做筛选(保留负值数轴往右+min)
            allClass_tocmpDistance[feature] + abs(min(allClass_tocmpDistance[feature]))

    #归一化第二次的距离作为相似度（贡献）   output_size=Norm([class_num, clients_num])
    allClass_normSimilarities = [[] for j in range(classes)]
    for feature in range(classes):
        class_normSimilarities = allClass_tocmpDistance[feature]
        # for client in range(clients):
        #     nums = clients_class_num[client]
        #     if len(nums) == 0:
        #         class_normSimilarities[client] = 0
        #     else:
        #         # class_normSimilarities[client] = class_normSimilarities[client] * nums[feature]      #查看一下公式是否需要num[feature]
        #         class_normSimilarities[client] = class_normSimilarities[client]      #查看一下公式是否需要num[feature]
        class_sum = sum(class_normSimilarities)
        for client in range(clients):
            if class_sum == 0:
                class_normSimilarities[client] = 0
            else:
                class_normSimilarities[client] = class_normSimilarities[client]/class_sum
        allClass_normSimilarities[feature] = class_normSimilarities


    return allClass_normSimilarities

#计算客户端和全局原型相似度
def client_similarity(preclient_features, clients_features, clients_class_num, classes, clients):
    #求原型之间的差值的二范数
    allClass_change = [[] for j in range(classes)]
    allClass_changeSum = [[] for j in range(classes)]
    for feature in range(classes):
        allClass_changeSum[feature] = 0
        curClass_change = [0 for i in range(clients)]
        for client in range(clients):
            if len(preclient_features[client]) == 0 and len(clients_features[client]) == 0:
                curClass_change[client] = 0
            elif len(preclient_features[client]) == 0:
                curClass_change[client] = torch.norm(clients_features[client] * (-1))
                allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
            elif len(clients_features[client]) == 0:
                curClass_change[client] = torch.norm(preclient_features[client])
                allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
            elif len(preclient_features[client]) != 0 and len(clients_features[client]) != 0:
                if preclient_features[client][feature] is None and clients_features[client][feature] is None:
                    curClass_change[client] = 0
                elif preclient_features[client][feature] is None:
                    curClass_change[client] = torch.norm(clients_features[client][feature])
                    allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
                elif clients_features[client][feature] is None:
                    curClass_change[client] = torch.norm(preclient_features[client][feature])
                    allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
                elif len(preclient_features[client][feature]) != 0 and len(clients_features[client][feature]) != 0:
                    curClass_change[client] = torch.norm(torch.sub(preclient_features[client][feature], clients_features[client][feature]))
                    allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
                else:
                    continue
            else:
                continue
            allClass_change[feature] = curClass_change


    classes_avg = [None for j in range(classes)]
    # 求每个类型的avg_feature
    for feature in range(classes):
        cnt = 0
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue

            client_feature = clients_features[client][feature]  #记录当前客户端的当前类

            if client_feature is None:
                continue
            if classes_avg[feature] is None:
                classes_avg[feature] = client_feature
                cnt += 1
            elif classes_avg[feature].shape == client_feature.shape:
                classes_avg[feature] = torch.add(classes_avg[feature], client_feature)   #将当前特征加入
                cnt += 1
        if classes_avg[feature] is not None:
            classes_avg[feature] = torch.div(classes_avg[feature], cnt)   #计算出来的平均特征

    #求每个客户端的每个类到全局该类平均特征的距离（余弦）
    allClass_distance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_distance = [0 for i in range(clients)]
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue
            if clients_features[client][feature] is None:
                continue
            if classes_avg[feature] is not None and classes_avg[feature].shape == clients_features[client][feature].shape:
                cosin_smilarity = torch.cosine_similarity(clients_features[client][feature].unsqueeze(0).float(), classes_avg[feature].unsqueeze(0).float())
                cosine_similarity = torch.mean(cosin_smilarity)
                if cosine_similarity.item() > 0:
                    curClass_distance[client] = cosine_similarity.item()
        allClass_distance[feature] = curClass_distance

    #求全局归一化计算后的余弦距离
    allClass_cmpDistance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_cmpDistance = [0 for i in range(clients)]
        curClass_sum = sum(allClass_distance[feature])      #计算当前类的原型距离之和
        for client in range(clients):
            if allClass_distance[feature][client] is None or allClass_distance[feature][client] == 0:
                continue
            elif allClass_distance[feature][client] is not None:
                curcosine_similarity = allClass_distance[feature][client]
                cosine_cmpSimilarity = (curcosine_similarity/curClass_sum) * (classes_avg[feature])    #归一化
                curClass_cmpDistance[client] = cosine_cmpSimilarity
        allClass_cmpDistance[feature] = curClass_cmpDistance

    # 求计算后每个类型的avg_feature
    classes_cmpAvg = [None for j in range(classes)]
    for feature in range(classes):
        cnt = 0
        for client in range(clients):
            if allClass_cmpDistance[feature][client] is None:
                continue

            cur_feature_client = allClass_cmpDistance[feature][client]  #记录当前客户端的当前类

            if cur_feature_client is None:
                continue

            if type(cur_feature_client) == int:
                continue

            if classes_cmpAvg[feature] is None:
                classes_cmpAvg[feature] = cur_feature_client
                cnt += 1
            elif classes_cmpAvg[feature].shape == cur_feature_client.shape:
                classes_cmpAvg[feature] = torch.add(classes_cmpAvg[feature], cur_feature_client)   #将当前特征加入
                cnt += 1
        if classes_cmpAvg[feature] is not None:
            classes_cmpAvg[feature] = torch.div(classes_cmpAvg[feature], cnt)   #计算出来的平均特征

    #计算客户端类原型到计算后的原型距离
    allClass_tocmpDistance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_tocmpDistance = [0 for i in range(clients)]
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue
            if clients_features[client][feature] is None:
                continue
            if classes_cmpAvg[feature] is not None and classes_cmpAvg[feature].shape == clients_features[client][feature].shape:
                cosine_similarity = torch.mean(torch.cosine_similarity(clients_features[client][feature].unsqueeze(0).float(), classes_cmpAvg[feature].unsqueeze(0).float()))
                if cosine_similarity.item() > 0:
                    curClass_tocmpDistance[client] = cosine_similarity.item()
        allClass_tocmpDistance[feature] = curClass_tocmpDistance

    #归一化第二次的距离作为相似度（贡献）
    allClass_normSimilarities = [[] for j in range(classes)]
    for feature in range(classes):
        class_normSimilarities = allClass_tocmpDistance[feature]
        for client in range(clients):
            nums = clients_class_num[client]
            if len(nums) == 0:
                class_normSimilarities[client] = 0
            else:
                class_normSimilarities[client] = class_normSimilarities[client] * nums[feature]      #查看一下公式是否需要num[feature]
                # class_normSimilarities[client] = class_normSimilarities[client]      #查看一下公式是否需要num[feature]
        class_sum = sum(class_normSimilarities)
        for client in range(clients):
            if class_sum == 0:
                class_normSimilarities[client] = 0
            else:
                class_normSimilarities[client] = class_normSimilarities[client]/class_sum
        allClass_normSimilarities[feature] = class_normSimilarities

    #归一化原型差值*相似度（贡献）
    allClass_normChange = [[] for j in range(classes)]
    allClass_normchangeSum = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_normChange = [0 for i in range(clients)]
        allClass_normchangeSum[feature] = 0
        for client in range(clients):
            if allClass_change[feature][client] is None or allClass_change[feature][client] == 0:
                continue
            elif allClass_change[feature][client] is not None:
                curfeature_client_change = allClass_change[feature][client]
                curfeature_client_normchange_tmp = curfeature_client_change
                curfeature_client_normchange = curfeature_client_normchange_tmp * allClass_normSimilarities[feature][client] # 二范数*贡献（范数再归一化一次）
                curClass_normChange[client] = curfeature_client_normchange.item()
                allClass_normchangeSum[feature] = torch.add(allClass_normchangeSum[feature], curClass_normChange[client])

        allClass_normChange[feature] = curClass_normChange

    #归一化最终乘积值
    allClass_final = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_final = [0 for i in range(clients)]
        for client in range(clients):
            if allClass_normChange[feature][client] is None or allClass_normChange[feature][client] == 0:
                continue
            elif allClass_normChange[feature][client] is not None:
                curfeature_client_normchange = allClass_normChange[feature][client]
                if (allClass_normChange[feature][client] != 0):
                    curfeature_client_normchange = curfeature_client_normchange / allClass_normchangeSum[feature] # 归一化
                    curClass_final[client] = curfeature_client_normchange.item()
                else:
                    continue
        allClass_final[feature] = curClass_final

    # return allClass_normSimilarities
    return allClass_final

#计算客户端和全局原型相似度(只有本轮差距)
def client_similarity_diff(preclient_features, clients_features, clients_class_num, classes, clients):
    classes_avg = [None for j in range(classes)]
    # 求每个类型的avg_feature
    for feature in range(classes):
        cnt = 0
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue

            client_feature = clients_features[client][feature]  #记录当前客户端的当前类

            if client_feature is None:
                continue
            if classes_avg[feature] is None:
                classes_avg[feature] = client_feature
                cnt += 1
            elif classes_avg[feature].shape == client_feature.shape:
                classes_avg[feature] = torch.add(classes_avg[feature], client_feature)   #将当前特征加入
                cnt += 1
        if classes_avg[feature] is not None:
            classes_avg[feature] = torch.div(classes_avg[feature], cnt)   #计算出来的平均特征

    #求每个客户端的每个类到全局该类平均特征的距离（余弦）
    allClass_distance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_distance = [0 for i in range(clients)]
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue
            if clients_features[client][feature] is None:
                continue
            if classes_avg[feature] is not None and classes_avg[feature].shape == clients_features[client][feature].shape:
                cosin_smilarity = torch.cosine_similarity(clients_features[client][feature].unsqueeze(0).float(), classes_avg[feature].unsqueeze(0).float())
                cosine_similarity = torch.mean(cosin_smilarity)
                if cosine_similarity.item() > 0:
                    curClass_distance[client] = cosine_similarity.item()
        allClass_distance[feature] = curClass_distance

    #求全局归一化计算后的余弦距离
    allClass_cmpDistance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_cmpDistance = [0 for i in range(clients)]
        curClass_sum = sum(allClass_distance[feature])      #计算当前类的原型距离之和
        for client in range(clients):
            if allClass_distance[feature][client] is None or allClass_distance[feature][client] == 0:
                continue
            elif allClass_distance[feature][client] is not None:
                curcosine_similarity = allClass_distance[feature][client]
                cosine_cmpSimilarity = (curcosine_similarity/curClass_sum) * (classes_avg[feature])    #归一化
                curClass_cmpDistance[client] = cosine_cmpSimilarity
        allClass_cmpDistance[feature] = curClass_cmpDistance

    # 求计算后每个类型的avg_feature
    classes_cmpAvg = [None for j in range(classes)]
    for feature in range(classes):
        cnt = 0
        for client in range(clients):
            if allClass_cmpDistance[feature][client] is None:
                continue

            cur_feature_client = allClass_cmpDistance[feature][client]  #记录当前客户端的当前类

            if cur_feature_client is None:
                continue

            if type(cur_feature_client) == int:
                continue

            if classes_cmpAvg[feature] is None:
                classes_cmpAvg[feature] = cur_feature_client
                cnt += 1
            elif classes_cmpAvg[feature].shape == cur_feature_client.shape:
                classes_cmpAvg[feature] = torch.add(classes_cmpAvg[feature], cur_feature_client)   #将当前特征加入
                cnt += 1
        if classes_cmpAvg[feature] is not None:
            classes_cmpAvg[feature] = torch.div(classes_cmpAvg[feature], cnt)   #计算出来的平均特征

    #计算客户端类原型到计算后的原型距离
    allClass_tocmpDistance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_tocmpDistance = [0 for i in range(clients)]
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue
            if clients_features[client][feature] is None:
                continue
            if classes_cmpAvg[feature] is not None and classes_cmpAvg[feature].shape == clients_features[client][feature].shape:
                cosine_similarity = torch.mean(torch.cosine_similarity(clients_features[client][feature].unsqueeze(0).float(), classes_cmpAvg[feature].unsqueeze(0).float()))
                if cosine_similarity.item() > 0:
                    curClass_tocmpDistance[client] = cosine_similarity.item()
        allClass_tocmpDistance[feature] = curClass_tocmpDistance

    #归一化第二次的距离作为相似度（贡献）
    allClass_normSimilarities = [[] for j in range(classes)]
    allClass_normSimilaritiesSum = [[] for j in range(classes)]
    for feature in range(classes):
        class_normSimilarities = allClass_tocmpDistance[feature]
        for client in range(clients):
            nums = clients_class_num[client]
            if len(nums) == 0:
                class_normSimilarities[client] = 0
            else:
                class_normSimilarities[client] = class_normSimilarities[client] * nums[feature]      #查看一下公式是否需要num[feature]
                # class_normSimilarities[client] = class_normSimilarities[client]      #查看一下公式是否需要num[feature]
        class_sum = sum(class_normSimilarities)
        allClass_normSimilaritiesSum[feature] = class_sum
        for client in range(clients):
            if class_sum == 0:
                class_normSimilarities[client] = 0
            else:
                class_normSimilarities[client] = class_normSimilarities[client]/class_sum
        allClass_normSimilarities[feature] = class_normSimilarities

    # #归一化原型差值*相似度（贡献）
    # allClass_normChange = [[] for j in range(classes)]
    # allClass_normchangeSum = [[] for j in range(classes)]
    # for feature in range(classes):
    #     curClass_normChange = [0 for i in range(clients)]
    #     allClass_normchangeSum[feature] = 0
    #     for client in range(clients):
    #         if allClass_change[feature][client] is None or allClass_change[feature][client] == 0:
    #             continue
    #         elif allClass_change[feature][client] is not None:
    #             curfeature_client_change = allClass_change[feature][client]
    #             curfeature_client_normchange_tmp = curfeature_client_change
    #             curfeature_client_normchange = curfeature_client_normchange_tmp * allClass_normSimilarities[feature][client] # 二范数*贡献（范数再归一化一次）
    #             curClass_normChange[client] = curfeature_client_normchange.item()
    #             allClass_normchangeSum[feature] = torch.add(allClass_normchangeSum[feature], curClass_normChange[client])
    #
    #     allClass_normChange[feature] = curClass_normChange

    #归一化最终乘积值
    allClass_final = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_final = [0 for i in range(clients)]
        for client in range(clients):
            if allClass_normSimilarities[feature][client] is None or allClass_normSimilarities[feature][client] == 0:
                continue
            elif allClass_normSimilarities[feature][client] is not None:
                curfeature_client_normchange = allClass_normSimilarities[feature][client]
                if (allClass_normSimilarities[feature][client] != 0):
                    curfeature_client_normchange = curfeature_client_normchange / allClass_normSimilaritiesSum[feature] # 归一化
                    curClass_final[client] = curfeature_client_normchange
                else:
                    continue
        allClass_final[feature] = curClass_final

    return allClass_normSimilarities




#计算客户端和全局原型相似度
def client_similarity_alone(preclient_features, clients_features, clients_class_num, classes, clients):
    #求原型之间的差值的二范数
    allClass_change = [[] for j in range(classes)]
    allClass_changeSum = [[] for j in range(classes)]
    for feature in range(classes):
        allClass_changeSum[feature] = 0
        curClass_change = [0 for i in range(clients)]
        for client in range(clients):
            if len(preclient_features[client]) == 0 and len(clients_features[client]) == 0:
                curClass_change[client] = 0
            elif len(preclient_features[client]) == 0:
                curClass_change[client] = torch.norm(clients_features[client] * (-1))
                allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
            elif len(clients_features[client]) == 0:
                curClass_change[client] = torch.norm(preclient_features[client])
                allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
            elif len(preclient_features[client]) != 0 and len(clients_features[client]) != 0:
                if preclient_features[client][feature] is None and clients_features[client][feature] is None:
                    curClass_change[client] = 0
                elif preclient_features[client][feature] is None:
                    curClass_change[client] = torch.norm(clients_features[client][feature])
                    allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
                elif clients_features[client][feature] is None:
                    curClass_change[client] = torch.norm(preclient_features[client][feature])
                    allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
                elif len(preclient_features[client][feature]) != 0 and len(clients_features[client][feature]) != 0:
                    curClass_change[client] = torch.norm(torch.sub(preclient_features[client][feature], clients_features[client][feature]))
                    allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
                else:
                    continue
            else:
                continue
            allClass_change[feature] = curClass_change


    classes_avg = [None for j in range(classes)]
    # 求每个类型的avg_feature
    for feature in range(classes):
        cnt = 0
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue

            client_feature = clients_features[client][feature]  #记录当前客户端的当前类

            if client_feature is None:
                continue
            if classes_avg[feature] is None:
                classes_avg[feature] = client_feature
                cnt += 1
            elif classes_avg[feature].shape == client_feature.shape:
                classes_avg[feature] = torch.add(classes_avg[feature], client_feature)   #将当前特征加入
                cnt += 1
        if classes_avg[feature] is not None:
            classes_avg[feature] = torch.div(classes_avg[feature], cnt)   #计算出来的平均特征

    #求每个客户端的每个类到全局该类平均特征的距离（余弦）
    allClass_distance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_distance = [0 for i in range(clients)]
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue
            if clients_features[client][feature] is None:
                continue
            if classes_avg[feature] is not None and classes_avg[feature].shape == clients_features[client][feature].shape:
                cosin_smilarity = torch.cosine_similarity(clients_features[client][feature].unsqueeze(0).float(), classes_avg[feature].unsqueeze(0).float())
                cosine_similarity = torch.mean(cosin_smilarity)
                if cosine_similarity.item() > 0:
                    curClass_distance[client] = cosine_similarity.item()
        allClass_distance[feature] = curClass_distance

    #求全局归一化计算后的余弦距离
    allClass_cmpDistance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_cmpDistance = [0 for i in range(clients)]
        curClass_sum = sum(allClass_distance[feature])      #计算当前类的原型距离之和
        for client in range(clients):
            if allClass_distance[feature][client] is None or allClass_distance[feature][client] == 0:
                continue
            elif allClass_distance[feature][client] is not None:
                curcosine_similarity = allClass_distance[feature][client]
                cosine_cmpSimilarity = (curcosine_similarity/curClass_sum) * (classes_avg[feature])    #归一化
                curClass_cmpDistance[client] = cosine_cmpSimilarity
        allClass_cmpDistance[feature] = curClass_cmpDistance

    # 求计算后每个类型的avg_feature
    classes_cmpAvg = [None for j in range(classes)]
    for feature in range(classes):
        cnt = 0
        for client in range(clients):
            if allClass_cmpDistance[feature][client] is None:
                continue

            cur_feature_client = allClass_cmpDistance[feature][client]  #记录当前客户端的当前类

            if cur_feature_client is None:
                continue

            if type(cur_feature_client) == int:
                continue

            if classes_cmpAvg[feature] is None:
                classes_cmpAvg[feature] = cur_feature_client
                cnt += 1
            elif classes_cmpAvg[feature].shape == cur_feature_client.shape:
                classes_cmpAvg[feature] = torch.add(classes_cmpAvg[feature], cur_feature_client)   #将当前特征加入
                cnt += 1
        if classes_cmpAvg[feature] is not None:
            classes_cmpAvg[feature] = torch.div(classes_cmpAvg[feature], cnt)   #计算出来的平均特征

    #计算客户端类原型到计算后的原型距离
    allClass_tocmpDistance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_tocmpDistance = [0 for i in range(clients)]
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue
            if clients_features[client][feature] is None:
                continue
            if classes_cmpAvg[feature] is not None and classes_cmpAvg[feature].shape == clients_features[client][feature].shape:
                cosine_similarity = torch.mean(torch.cosine_similarity(clients_features[client][feature].unsqueeze(0).float(), classes_cmpAvg[feature].unsqueeze(0).float()))
                if cosine_similarity.item() > 0:
                    curClass_tocmpDistance[client] = cosine_similarity.item()
        allClass_tocmpDistance[feature] = curClass_tocmpDistance

    #归一化第二次的距离作为相似度（贡献）
    allClass_normSimilarities = [[] for j in range(classes)]
    for feature in range(classes):
        class_normSimilarities = allClass_tocmpDistance[feature]
        for client in range(clients):
            nums = clients_class_num[client]
            if len(nums) == 0:
                class_normSimilarities[client] = 0
            else:
                class_normSimilarities[client] = class_normSimilarities[client] * nums[feature]
        class_sum = sum(class_normSimilarities)
        for client in range(clients):
            if class_sum == 0:
                class_normSimilarities[client] = 0
            else:
                class_normSimilarities[client] = class_normSimilarities[client]/class_sum
        allClass_normSimilarities[feature] = class_normSimilarities

    #归一化原型差值*相似度（贡献）
    allClass_normChange = [[] for j in range(classes)]
    allClass_normchangeSum = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_normChange = [0 for i in range(clients)]
        allClass_normchangeSum[feature] = 0
        for client in range(clients):
            if allClass_change[feature][client] is None or allClass_change[feature][client] == 0:
                continue
            elif allClass_change[feature][client] is not None:
                curfeature_client_change = allClass_change[feature][client]
                curfeature_client_normchange_tmp = curfeature_client_change
                curfeature_client_normchange = allClass_normSimilarities[feature][client] # 二范数*贡献（范数再归一化一次）【去掉变化速率的影响，只看相似度】
                curClass_normChange[client] = curfeature_client_normchange
                allClass_normchangeSum[feature] = torch.add(allClass_normchangeSum[feature], curClass_normChange[client])

        allClass_normChange[feature] = curClass_normChange

    #归一化最终乘积值
    allClass_final = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_final = [0 for i in range(clients)]
        for client in range(clients):
            if allClass_normChange[feature][client] is None or allClass_normChange[feature][client] == 0:
                continue
            elif allClass_normChange[feature][client] is not None:
                curfeature_client_normchange = allClass_normChange[feature][client]
                if (allClass_normChange[feature][client] != 0):
                    curfeature_client_normchange = curfeature_client_normchange / allClass_normchangeSum[feature] # 归一化
                    curClass_final[client] = curfeature_client_normchange.item()
                else:
                    continue
        allClass_final[feature] = curClass_final

    # return allClass_normSimilarities
    return allClass_final

#计算客户端和全局原型相似度(前轮平均原型改进)
def client_similarity_prodiff(preclient_features, clients_features, clients_class_num, classes, clients):
    #计算前一轮平均类原型
    pre_classes_avg = [None for j in range(classes)]
    # 求每个类型的avg_feature
    for feature in range(classes):
        cnt = 0
        for client in range(clients):
            if preclient_features[client] is None or len(preclient_features[client]) == 0:
                continue
            client_feature = preclient_features[client][feature]  #记录当前客户端的当前类
            if client_feature is None:
                continue
            if pre_classes_avg[feature] is None:
                pre_classes_avg[feature] = client_feature
                cnt += 1
            elif pre_classes_avg[feature].shape == client_feature.shape:
                pre_classes_avg[feature] = torch.add(pre_classes_avg[feature], client_feature)   #将当前特征加入
                cnt += 1
        if pre_classes_avg[feature] is not None:
            pre_classes_avg[feature] = torch.div(pre_classes_avg[feature], cnt)   #计算出来的平均特征

    #计算当前客户端类原型和前一轮平均类原型之间的差值的二范数（p(t-1)_avg - pt）
    allClass_change = [[] for j in range(classes)]
    allClass_changeSum = [[] for j in range(classes)]
    for feature in range(classes):
        allClass_changeSum[feature] = 0
        curClass_change = [0 for i in range(clients)]
        for client in range(clients):
            if pre_classes_avg[feature] is None:
                if len(clients_features[client]) == 0:
                    curClass_change[client] = 0
                else:
                    if clients_features[client][feature] is None:
                        curClass_change[client] = 0
                    else:
                        curClass_change[client] = torch.norm(clients_features[client][feature])
                        allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
            else:
                if len(clients_features[client]) == 0:
                    curClass_change[client] = torch.norm(pre_classes_avg[feature])
                    allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
                else:
                    if clients_features[client][feature] is None:
                        curClass_change[client] = torch.norm(pre_classes_avg[feature])
                        allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
                    else:
                        curClass_change[client] = torch.norm(
                            torch.sub(pre_classes_avg[feature], clients_features[client][feature]))
                        allClass_changeSum[feature] = torch.add(allClass_changeSum[feature], curClass_change[client])
            allClass_change[feature] = curClass_change
            #(allchass_change归一化)


    classes_avg = [None for j in range(classes)]
    # 求每个类型的avg_feature
    for feature in range(classes):
        cnt = 0
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue

            client_feature = clients_features[client][feature]  #记录当前客户端的当前类

            if client_feature is None:
                continue
            if classes_avg[feature] is None:
                classes_avg[feature] = client_feature
                cnt += 1
            elif classes_avg[feature].shape == client_feature.shape:
                classes_avg[feature] = torch.add(classes_avg[feature], client_feature)   #将当前特征加入
                cnt += 1
        if classes_avg[feature] is not None:
            classes_avg[feature] = torch.div(classes_avg[feature], cnt)   #计算出来的平均特征

    #求每个客户端的每个类到全局该类平均特征的距离（余弦）
    allClass_distance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_distance = [0 for i in range(clients)]
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue
            if clients_features[client][feature] is None:
                continue
            if classes_avg[feature] is not None and classes_avg[feature].shape == clients_features[client][feature].shape:
                cosin_smilarity = torch.cosine_similarity(clients_features[client][feature].unsqueeze(0).float(), classes_avg[feature].unsqueeze(0).float())
                cosine_similarity = torch.mean(cosin_smilarity)
                if cosine_similarity.item() > 0:
                    curClass_distance[client] = cosine_similarity.item()
        allClass_distance[feature] = curClass_distance

    #求全局归一化计算后的余弦距离
    allClass_cmpDistance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_cmpDistance = [0 for i in range(clients)]
        curClass_sum = sum(allClass_distance[feature])      #计算当前类的原型距离之和
        for client in range(clients):
            if allClass_distance[feature][client] is None or allClass_distance[feature][client] == 0:
                continue
            elif allClass_distance[feature][client] is not None:
                curcosine_similarity = allClass_distance[feature][client]
                cosine_cmpSimilarity = (curcosine_similarity/curClass_sum) * (classes_avg[feature])    #归一化
                curClass_cmpDistance[client] = cosine_cmpSimilarity
        allClass_cmpDistance[feature] = curClass_cmpDistance

    # 求计算后每个类型的avg_feature
    classes_cmpAvg = [None for j in range(classes)]
    for feature in range(classes):
        cnt = 0
        for client in range(clients):
            if allClass_cmpDistance[feature][client] is None:
                continue

            cur_feature_client = allClass_cmpDistance[feature][client]  #记录当前客户端的当前类

            if cur_feature_client is None:
                continue

            if type(cur_feature_client) == int:
                continue

            if classes_cmpAvg[feature] is None:
                classes_cmpAvg[feature] = cur_feature_client
                cnt += 1
            elif classes_cmpAvg[feature].shape == cur_feature_client.shape:
                classes_cmpAvg[feature] = torch.add(classes_cmpAvg[feature], cur_feature_client)   #将当前特征加入
                cnt += 1
        if classes_cmpAvg[feature] is not None:
            classes_cmpAvg[feature] = torch.div(classes_cmpAvg[feature], cnt)   #计算出来的平均特征

    #计算客户端类原型到计算后的原型距离
    allClass_tocmpDistance = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_tocmpDistance = [0 for i in range(clients)]
        for client in range(clients):
            if clients_features[client] is None or len(clients_features[client]) == 0:
                continue
            if clients_features[client][feature] is None:
                continue
            if classes_cmpAvg[feature] is not None and classes_cmpAvg[feature].shape == clients_features[client][feature].shape:
                cosine_similarity = torch.mean(torch.cosine_similarity(clients_features[client][feature].unsqueeze(0).float(), classes_cmpAvg[feature].unsqueeze(0).float()))
                if cosine_similarity.item() > 0:
                    curClass_tocmpDistance[client] = cosine_similarity.item()
        allClass_tocmpDistance[feature] = curClass_tocmpDistance

    #归一化第二次的距离作为相似度（贡献）
    allClass_normSimilarities = [[] for j in range(classes)]
    for feature in range(classes):
        class_normSimilarities = allClass_tocmpDistance[feature]
        for client in range(clients):
            nums = clients_class_num[client]
            if len(nums) == 0:
                class_normSimilarities[client] = 0
            else:
                # class_normSimilarities[client] = class_normSimilarities[client] * nums[feature]      #查看一下公式是否需要num[feature]
                class_normSimilarities[client] = class_normSimilarities[client]      #查看一下公式是否需要num[feature]
        class_sum = sum(class_normSimilarities)
        for client in range(clients):
            if class_sum == 0:
                class_normSimilarities[client] = 0
            else:
                class_normSimilarities[client] = class_normSimilarities[client]/class_sum
        allClass_normSimilarities[feature] = class_normSimilarities

    #归一化原型差值*相似度（贡献）
    allClass_normChange = [[] for j in range(classes)]
    allClass_normchangeSum = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_normChange = [0 for i in range(clients)]
        allClass_normchangeSum[feature] = 0
        for client in range(clients):
            if allClass_change[feature][client] is None or allClass_change[feature][client] == 0:
                continue
            elif allClass_change[feature][client] is not None:
                curfeature_client_change = allClass_change[feature][client]
                curfeature_client_normchange_tmp = curfeature_client_change
                allClass_normSimilarities_tmp = allClass_normSimilarities[feature][client]
                curfeature_client_normchange = curfeature_client_normchange_tmp * allClass_normSimilarities_tmp # 二范数*贡献（范数再归一化一次）
                curClass_normChange[client] = curfeature_client_normchange.item()
                allClass_normchangeSum[feature] = torch.add(allClass_normchangeSum[feature], curClass_normChange[client])

        allClass_normChange[feature] = curClass_normChange

    #归一化最终乘积值
    allClass_final = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_final = [0 for i in range(clients)]
        for client in range(clients):
            if allClass_normChange[feature][client] is None or allClass_normChange[feature][client] == 0:
                continue
            elif allClass_normChange[feature][client] is not None:
                curfeature_client_normchange = allClass_normChange[feature][client]
                if (allClass_normChange[feature][client] != 0):
                    curfeature_client_normchange = curfeature_client_normchange / allClass_normchangeSum[feature] # 归一化
                    curClass_final[client] = curfeature_client_normchange.item()
                else:
                    continue
        allClass_final[feature] = curClass_final

    return allClass_final



def cal_client_full_contri(all_similarities, classes, clients):   #求客户端贡献比例（后续按类权重设置个参数）
    p = [0 for i in range(clients)]
    for client in range(clients):
        for tag in range(classes):
            if all_similarities[tag][client] is None:
                continue
            # classes = 10, / 20, classes = 100, / 200
            p[client] += all_similarities[tag][client]
    # print("sum p is:", str(sum(p)))
    p_sum = sum(p)
    for client in range(clients):
        p[client] = p[client]/p_sum
    return p      #p归一化

def cal_prototypes_full_contri(all_similarities, clients_features, classes, clients):   #求客户端贡献比例（后续按类权重设置个参数）
    allClass_prototype = [[] for j in range(classes)]
    for feature in range(classes):
        curClass_prototype = [0 for i in range(clients)]
        for client in range(clients):
            if len(clients_features[client]) == 0:
                continue
            else:
                if clients_features[client][feature] is not None:
                    curClass_prototype_tmp = all_similarities[feature][client]
                    client_features_tmp = clients_features[client][feature]
                    weight_prototype = curClass_prototype_tmp * client_features_tmp   #weight * prototype
                    curClass_prototype[client] = weight_prototype
        allClass_prototype[feature] = curClass_prototype

    classes_avg = [None for j in range(classes)]
    # 求每个类型的avg_feature
    for feature in range(classes):
        cnt = 0
        for client in range(clients):

            feature_clients = allClass_prototype[feature][client]  #记录当前客户端的当前类

            if type(feature_clients) is int:
                continue
            else:
                if classes_avg[feature] is None:
                    classes_avg[feature] = feature_clients
                    cnt += 1
                else:
                    classes_avg[feature] = torch.add(classes_avg[feature], feature_clients)   #将当前特征加入
                    cnt += 1
        # if classes_avg[feature] is not None:
        #     classes_avg[feature] = torch.div(classes_avg[feature], cnt)   #计算出来的平均特征
    return classes_avg

def cal_client_contribution(global_model, client_num, local_models):
    # 计算每个客户端模型贡献度
    cur_global_model = copy.deepcopy(global_model)
    Allclient_contribution = []
    sum_contribution = 0

    for client in range(client_num):
        if client in local_models:
            diff = cur_global_model - local_models[client]  # compute the diff of global to local models
            diff_norm = diff.norm().item()
            Allclient_contribution.append(diff_norm)
            sum_contribution += diff_norm
        else:
            Allclient_contribution.append(0)

    for client, contribution in enumerate(Allclient_contribution):
        tmp = Allclient_contribution[client]
        cur_contribution = tmp / sum_contribution
        Allclient_contribution[client] = cur_contribution

    return Allclient_contribution

def Normalize(array):
    '''
    Normalize the array
    '''
    mx = np.nanmax(array)
    mn = np.nanmin(array)
    t = (array-mn)/(mx-mn)
    return t,mx,mn


def save_mydataset(name):
    # data = np.load('/data/sharedata/flceh/Cifar10_IID_numc50_alpha0_seed0.npy', allow_pickle=True)
    fileload = '/data/sharedata/flceh/'
    filename = fileload + name + '.npy'
    data = np.load(filename, allow_pickle=True)
    train_spilt_pro = 0.8

    data_list = data.tolist()

    clients = {}

    print(type(data_list))

    test_xs_pre = data_list[len(data_list) - 1].get('test_data')
    test_xs = []
    for i in range(len(test_xs_pre)):
        image_tensor = test_xs_pre[i]
        transposed_image_tensor = image_tensor.transpose(2, 0, 1)
        test_xs.append(transposed_image_tensor)
    test_xs = np.array(test_xs)
    test_ys = data_list[len(data_list) - 1].get('test_targets')

    for i in range(len(data_list) - 1):
        tmp = {}
        xs = []
        xs_pre = data_list[i].get('train_data')
        for j in range(len(xs_pre)):
            image_tensor = xs_pre[j]
            transposed_image_tensor = image_tensor.transpose(2, 0, 1)
            xs.append(transposed_image_tensor)

        xs = np.array(xs)
        ys = data_list[i].get('train_targets')

        train_nums = int(len(xs) * train_spilt_pro)

        tmp['train_xs'] = xs[:train_nums]
        tmp['train_ys'] = ys[:train_nums]
        tmp['test_xs'] = xs[train_nums:]
        tmp['test_ys'] = ys[train_nums:]
        clients[i] = tmp

    res = {}
    res['clients'] = clients
    res['test_xs'] = test_xs
    res['test_ys'] = test_ys

    np.save("/data/yaominghao/code/FedRepo/datasets/mydataset/" + name + '.npy', res)
    print(name + "save success!")

    return

#数据集本身取了每个客户端80%的数据集作为训练集
def load_mydataset(filename):
    if 'FASHIONMNIST' in filename:
        fileload = '/data/yaominghao/data/newdata/FashionMINIST/'
    elif 'Cifar10' in filename:
        fileload = '/data/yaominghao/code/FedRepo/datasets/mydataset/'
    else:
        fileload = ''

    filename = fileload + filename + '.npy'
    data = np.load(filename, allow_pickle=True)

    data_list = data.tolist()

    clients = data_list['clients']
    test_xs = data_list['test_xs']
    test_ys = data_list['test_ys']
    return clients, test_xs, test_ys

def load_Allmydataset_tmp(filename):
    if 'unbalanced' in filename:
        fileload = '/data/yaominghao/code/FedRepo/datasets/unbalanced_mydataset/'
    else:
        fileload = '/data/sharedata/flceh/'
    filename = fileload + filename + '.npy'
    data = np.load(filename, allow_pickle=True)

    data_list = data.tolist()
    clients_data = {}

    for i in range(50):
        cur_clients_data = {}
        cur_length = len(data_list[i]['train_data'])
        split_length = int(cur_length * 0.8)

        cur_clients_reshape_data = np.transpose(data_list[i]['train_data'], (0, 3, 1, 2))     #调整图片通道契合模型

        cur_clients_data['train_xs'] = cur_clients_reshape_data
        cur_clients_data['train_ys'] = data_list[i]['train_targets']
        cur_clients_data['test_xs'] = cur_clients_reshape_data[split_length:]
        cur_clients_data['test_ys'] = data_list[i]['train_targets'][split_length:]
        clients_data[i] = cur_clients_data

        # for j in range(100):
        #     p = clients_data['train_xs'][j]
        #     p = np.transpose(p, (1, 2, 0))
        #     plt.imshow(p)
        #     plt.show()

    test_xs = np.transpose(data_list[50]['test_data'], (0, 3, 1, 2))
    test_ys = data_list[50]['test_targets']
    return clients_data, test_xs, test_ys

def load_Cifar_Allmydataset(filename):
    if 'unbalanced' in filename:
        fileload = '/data/sharedata/flceh_unbalanced/'
    else:
        fileload = '/data/sharedata/flceh/'
    filename = fileload + filename + '.npy'
    data = np.load(filename, allow_pickle=True)

    data_list = data.tolist()
    clients_data = {}

    clients_instance = []
    for i in range(50):
        cur_clients_data = {}
        cur_length = len(data_list[i]['train_data'])
        split_length = int(cur_length * 0.8)

        cur_clients_reshape_data = np.transpose(data_list[i]['train_data'], (0, 3, 1, 2))     #调整图片通道契合模型

        cur_clients_data['train_xs'] = cur_clients_reshape_data        #训练数据取全部
        cur_clients_data['train_ys'] = data_list[i]['train_targets']
        cur_clients_data['test_xs'] = cur_clients_reshape_data[split_length:]
        cur_clients_data['test_ys'] = data_list[i]['train_targets'][split_length:]
        clients_data[i] = cur_clients_data

        # for i in range(100):
        #     p = cur_clients_reshape_data[i]
        #     p = np.transpose(p, (1, 2, 0))
        #     plt.imshow(p)
        #     plt.show()

        clients_instance.append(get_class_instance(cur_clients_data['train_ys']))

    # np.save(
    #     "/data/yaominghao/logs_2024/log202403/ClassInstance" + '_Cifar100_unalanced_Alldata' + '_NonIID0.5' + '.npy',
    #     clients_instance)

    cur_test_length = len(data_list[50]['test_data'])
    test_xs = np.transpose(data_list[50]['test_data'], (0, 3, 1, 2))
    test_ys = data_list[50]['test_targets']
    return clients_data, test_xs, test_ys

def load_Cifar_mydataset(filename):
    if 'unbalanced' in filename:
        fileload = '/data/sharedata/flceh_unbalanced/'
    else:
        fileload = '/data/sharedata/flceh/'
    filename = fileload + filename + '.npy'
    data = np.load(filename, allow_pickle=True)

    data_list = data.tolist()
    clients_data = {}

    clients_instance = []
    for i in range(50):
        cur_clients_data = {}
        cur_length = len(data_list[i]['train_data'])

        cur_clients_reshape_data = np.transpose(data_list[i]['train_data'], (0, 3, 1, 2))

        split_length = int(cur_length * 0.8)
        cur_clients_data['train_xs'] = cur_clients_reshape_data[:split_length]       #训练数据取部分
        cur_clients_data['train_ys'] = data_list[i]['train_targets'][:split_length]
        cur_clients_data['test_xs'] = cur_clients_reshape_data[split_length:]
        cur_clients_data['test_ys'] = data_list[i]['train_targets'][split_length:]

        clients_data[i] = cur_clients_data

        clients_instance.append(get_class_instance(cur_clients_data['train_ys']))

    # np.save(
    #     "/data/yaominghao/logs_2024/log202403/ClassInstance_"  + '_Cifar10' + '_NonIID0.5' + '.npy',
    #     clients_instance)

    cur_test_length = len(data_list[50]['test_data'])
    test_xs = np.transpose(data_list[50]['test_data'], (0, 3, 1, 2))
    test_ys = data_list[50]['test_targets']
    return clients_data, test_xs, test_ys

#取每个客户端全部的数据集作为训练集
def load_Allmydataset(filename):
    fileload = '/data/yaominghao/code/FedRepo/datasets/Allmydataset/'
    filename = fileload + filename + '.npy'
    data = np.load(filename, allow_pickle=True)

    data_list = data.tolist()
    clients_data = {}

    for i in range(50):
        cur_clients_data = {}
        cur_length = len(data_list[i]['train_data'])
        cur_clients_data['train_xs'] = data_list[i]['train_data'].reshape(cur_length, 3, 32, 32)
        cur_clients_data['train_ys'] = data_list[i]['train_targets']
        cur_clients_data['test_xs'] = data_list[i]['train_data'][:200]
        cur_clients_data['test_ys'] = data_list[i]['train_targets'][:200]
        clients_data[i] = cur_clients_data

    cur_test_length = len(data_list[50]['test_data'])
    test_xs = data_list[50]['test_data'].reshape(cur_test_length, 3, 32, 32)
    test_ys = data_list[50]['test_targets']
    return clients_data, test_xs, test_ys

def contribution_matrix_completion(filename):                 #补全贡献矩阵
    fileload = '/data/yaominghao/code/FedRepo/result/'
    file = fileload + filename + '.npy'
    contribution_matrix = torch.FloatTensor(
        np.load(file, allow_pickle=True))

    final = []
    for i in range(len(contribution_matrix)):
        cur = contribution_matrix[0]
        # print(cur)

        # 使用PyTorch中的SVD函数进行矩阵分解
        U, S, V = torch.svd(cur)

        # 重构原始矩阵
        reconstructed_A = U @ torch.diag(S) @ V.t()

        print("round ", i)

        final.append(reconstructed_A.tolist())

    competion_matrix_file = fileload +'final/' + filename + '_final.npy'
    np.save(competion_matrix_file, final)


def flatten(grad_update):
    flatten_grad = torch.cat([param.view(-1) for param in grad_update.state_dict().values()])
    return flatten_grad

def unflatten(flattened, normal_model):
    # 恢复成原来的形状
    restored_params = []
    start_idx = 0

    for param in normal_model.parameters():
        param_size = torch.prod(torch.tensor(param.size()))
        restored_params.append(flattened[start_idx:start_idx + param_size].view(param.size()))
        start_idx += param_size

    # 将恢复的参数设置回模型
    for param, restored_param in zip(normal_model.parameters(), restored_params):
        param.data = restored_param

    return normal_model

    # 打印恢复后的张量形状
    # print("Restored params shape:", restored_params[0].shape)
    # param_state_dict = {}
    # for name, param in normal_shape.state_dict().items():
    #     n_params = len(param.view(-1))
    #     value = torch.as_tensor(flattened[:n_params]).reshape(param.size())
    #     param_state_dict[name] = value
    #     gradient_flatten = flattened[n_params:]
    #
    # normal_shape.load_state_dict(param_state_dict, strict=False)
    # return normal_shape
    # unflattened_model = copy.deepcopy(normal_shape)
    # for name, param in unflattened_model.state_dict().items():
    #     n_params = len(param.view(-1))
    #     value = torch.as_tensor(flattened[:n_params]).reshape(param.size())
    #     unflattened_model[name] = value
    #     flattened = flattened[n_params:]
    # return unflattened_model

def get_class_instance(label_list):
        unique_elements, counts = np.unique(label_list, return_counts=True)
        instance_dict = dict(zip(list(unique_elements), list(counts)))
        class_instance = []
        for i in range(100):
            if i in instance_dict.keys():
               class_instance.append(instance_dict[i])
            else:
                class_instance.append(0)

        return class_instance