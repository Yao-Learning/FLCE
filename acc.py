import numpy as np
import matplotlib.pyplot as plt

def getLogData(filename):
    file_road = '/data/yaominghao/code/FedRepo/analysis/logs/'
    filename = file_road + filename + '.log'

    args =["Args(dataset='cifar10', split='dirichlet', dir_alpha=1.0, n_classes=10, n_clients=100, n_max_sam=None, "
           "c_ratio=0.1, net=None, max_round=1000, test_round=10, local_epochs=5, local_steps=None, batch_size=64, "
           "optimizer='SGD', lr=0.03, momentum=0.9, weight_decay=1e-05, max_grad_norm=100.0, cuda=True, algo='fedproavg', "
           "n_layer=8, none='none', fname='cifar10-K100-Dir-1.0-VGG8-fedproavg-Matrix.log')"]


    # 读取txt文件，以二维列表形式输出，每一个元素为一行
    file = open(filename, mode='r', encoding='UTF-8')
    admin = []
    rounds = []
    losses = []
    glo_taccs = []
    local_taccs = []
    contribution_matrix = []
    # 读取所有行(直到结束符 EOF)并返回列表
    contents = file.readlines()
    # print(contents)
    for msg in contents:
        # 删除结尾的\n字符
        msg = msg.strip('\n')
        # 字符串根据空格进行分割
        adm = msg.split(' ')
        admin.append(adm)
    file.close()

    # print(admin[0])
    # print(args)
    # if admin[0] != args:
    #     print("input error")
    for k in range(1, len(admin)):
        list = admin[k]
        for i in range(1, len(admin[k])):
            if(list[0] == '[ROUNDS]:'):
                rounds.append(int(list[i]))
                continue
            if (list[0] == '[LOSSES]:'):
                losses.append(float(list[i]))
                continue
            if(list[0] == '[GLO_TACCS]:'):
                glo_taccs.append(float(list[i]))
                continue
            if(list[0] == '[LOCAL_TACCS]:'):
                local_taccs.append(float(list[i]))
                continue
            if(list[0] == '[Contribution Matrix]:'):
                contribution_matrix.append(list[i])
                continue

    return rounds,losses,glo_taccs,local_taccs

def getLogGlobalPicture(*filename, Type):
    plt.figure(figsize=(5,5))
    plt.rcParams["font.family"] = 'Times New Roman'
    lw = 0.9
    fs = 13
    plt.xlabel('Communication Rounds', fontsize=fs)
    plt.ylabel('ACC',fontsize=fs)
    plt.grid(linestyle='-.')

    dict = {}
    for i in range(len(filename)):
        split_parts = filename[i].split('_')

        dict['dataset'] = split_parts[0]
        dict['datasetType'] = split_parts[1]
        dict['alpha'] = split_parts[3]
        if len(split_parts) <= 5:
            tmp = split_parts[-1].split('-')[-2]
            if tmp == 'fedavg':
                dict['alg'] = 'FedAvg'
            elif tmp == 'fedfa':
                dict['alg'] = 'FedFa'
            elif tmp == 'fedfv':
                dict['alg'] = 'FedFV'
            elif tmp == '1000':
                dict['alg'] = 'FedSV'
            elif tmp == 'fedprox':
                dict['alg'] = 'Fedprox'
            else:
                dict['alg'] = 'Fedproavg'
        else:
            dict['gaussian'] = split_parts[4]
            dict['addmod'] = split_parts[5]
            dict['duality'] = split_parts[6]
            tmp = split_parts[-1].split('-')[-2]
            if tmp == '1000':
                dict['alg'] = 'FLBS'
            elif tmp == 'fedavg':
                dict['alg'] = 'FedAvg'
            elif tmp == 'fedfa':
                dict['alg'] = 'FedFa'
            elif tmp == 'fedfv':
                dict['alg'] = 'FedFV'
            elif tmp == 'fedprox':
                dict['alg'] = 'Fedprox'
            else:
                dict['alg'] = tmp

        rounds, loss, glo_accs, local_accs = getLogData(filename[i])
        # glo_accs = glo_accs[:50]
        x_label = np.arange(0, len(glo_accs) * 10, 10)
        max_value = max(glo_accs)
        plt.plot(x_label, glo_accs, label=dict[Type] + ' ' + str("{:.3f}".format(round(max_value, 3))), linewidth=lw)

    key_str = ''
    for k,v in dict.items():
        if k == Type:
            continue
        else:
            key_str = key_str + '-' + v
    plt.title('global' + key_str, fontsize=10)
    # plt.title('Non-IID_Noise', fontsize=16)
    plt.legend()
    plt.show()


def getLogLocalPicture(*filename, Type):
    plt.figure(figsize=(5,5))
    lw = 0.9
    fs = 13
    plt.xlabel('Communication Rounds',fontsize=fs)
    plt.ylabel('ACC',fontsize=fs)
    plt.grid(linestyle='-.')

    dict = {}
    for i in range(len(filename)):
        split_parts = filename[i].split('_')

        dict['dataset'] = split_parts[0]
        dict['datasetType'] = split_parts[1]
        dict['alpha'] = split_parts[3]
        if len(split_parts) <= 5:
            dict['alg'] = split_parts[-1].split('-')[-2]
        else:
            dict['gaussian'] = split_parts[4]
            dict['addmod'] = split_parts[5]
            dict['duality'] = split_parts[6]
            dict['alg'] = split_parts[-1].split('-')[-2]

        rounds, loss, glo_accs,local_accs = getLogData(filename[i])
        x_label = np.arange(0, len(local_accs) * 5, 5)
        max_value = max(local_accs)
        plt.plot(x_label, local_accs, label=dict[Type]+str(max_value), linewidth=lw)

    key_str = ''
    for k,v in dict.items():
        if k == Type:
            continue
        else:
            key_str = key_str + '-' + v
    plt.title('local' + key_str, fontsize=10)

    plt.legend()
    plt.show()


if __name__ == '__main__':

    "IID-alpha0"
    file_IID_0_pro = 'Cifar10_IID_numc50_alpha0_seed0-K100-Dir-1.0-Res20-fedproavg-1000'
    file_IID_0_pro2 = 'Cifar10_IID_numc50_alpha0_seed0-K100-Dir-1.0-Res20-fedproavg2-1000'
    file_IID_0_avg = 'Cifar10_IID_numc50_alpha0_seed0-K100-Dir-1.0-Res20-fedavg-1000'
    file_IID_0_fa = 'Cifar10_IID_numc50_alpha0_seed0-K100-Dir-1.0-Res20-fedfa-1000'
    file_IID_0_fv = 'Cifar10_IID_numc50_alpha0_seed0-K100-Dir-1.0-Res20-fedfv-1000'
    file_IID_0_prox = 'Cifar10_IID_numc50_alpha0_seed0-K100-Dir-1.0-Res20-fedprox-1000'
    file_IID_0_svBig = 'Cifar10_IID_numc50_alpha0_seed0-K100-Dir-1.0-Res20-fedsharplyavg-1000-big'

    "IID-alpha0-gaussian0-addmod0.4-dualityFalse"
    file_IID_0_0_04_F_pro = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedproavg-1000'
    file_IID_0_0_04_F_avg = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedavg-1000'
    file_IID_0_0_04_F_fa = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedfa-1000'
    file_IID_0_0_04_F_fv = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedfv-1000'
    file_IID_0_0_04_F_prox = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedprox-1000'
    file_IID_0_0_04_F_svBig = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedsharplyavg-1000-big'

    "IID-alpha0-gaussian0.4-addmod0-dualityFalse"
    file_IID_0_04_0_F_pro = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedproavg-1000'
    file_IID_0_04_0_F_avg = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedavg-1000'
    file_IID_0_04_0_F_fa = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedfa-1000'
    file_IID_0_04_0_F_fv = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedfv-1000'
    file_IID_0_04_0_F_prox = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedprox-1000'
    file_IID_0_04_0_F_svBig = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedsharplyavg-1000-big'

    "IID-alpha0-gaussian0.2-addmod0.2-dualityFalse"
    file_IID_0_02_02_F_pro = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedproavg-1000'
    file_IID_0_02_02_F_avg = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedavg-1000'
    file_IID_0_02_02_F_fa = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedfa-1000'
    file_IID_0_02_02_F_fv = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedfv-1000'
    file_IID_0_02_02_F_prox = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedprox-1000'
    file_IID_0_02_02_F_svBig = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedsharplyavg-1000-big'

    "IID-alpha0-gaussian0.4-addmod0.4-dualityTrue"
    file_IID_0_04_04_T_pro = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedproavg-1000'
    file_IID_0_04_04_T_avg = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedavg-1000'
    file_IID_0_04_04_T_fa = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedfa-1000'
    file_IID_0_04_04_T_fv = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedfv-1000'
    file_IID_0_04_04_T_prox = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedprox-1000'
    file_IID_0_04_04_T_svBig = 'Cifar10_IID_numc50_alpha0_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedsharplyavg-1000-big'

    "NonIID-alpha01"
    file_NonIID_01_pro = 'Cifar10_NonIID_numc50_alpha0.1_seed0-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_01_avg = 'Cifar10_NonIID_numc50_alpha0.1_seed0-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_01_fa = 'Cifar10_NonIID_numc50_alpha0.1_seed0-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_01_fv = 'Cifar10_NonIID_numc50_alpha0.1_seed0-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_01_prox = 'Cifar10_NonIID_numc50_alpha0.1_seed0-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_01_sv = 'Cifar10_NonIID_numc50_alpha0.1_seed0-K100-Dir-1.0-Res20-fedsharplyavg-1000'

    "NonIID-alpha0.1-gaussian0-addmod0.4-dualityFalse"
    file_NonIID_01_0_04_F_pro = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_01_0_04_F_avg = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_01_0_04_F_fa = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_01_0_04_F_fv = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_01_0_04_F_prox = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_01_0_04_F_sv = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedsharplyavg-1000'

    "NonIID-alpha0.1-gaussian0.4-addmod0-dualityFalse"
    file_NonIID_01_04_0_F_pro = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_01_04_0_F_avg = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_01_04_0_F_fa = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_01_04_0_F_fv = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_01_04_0_F_prox = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_01_04_0_F_sv = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedsharplyavg-1000'

    "NonIID-alpha0.1-gaussian0.2-addmod0.2-dualityFalse"
    file_NonIID_01_02_02_F_pro = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_01_02_02_F_avg = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_01_02_02_F_fa = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_01_02_02_F_fv = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_01_02_02_F_prox = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_01_02_02_F_sv = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedsharplyavg-1000'

    "NonIID-alpha0.1-gaussian0.4-addmod0.4-dualityTrue"
    file_NonIID_01_04_04_T_pro = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_01_04_04_T_avg = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_01_04_04_T_fa = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_01_04_04_T_fv = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_01_04_04_T_prox = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_01_04_04_T_sv = 'Cifar10_NonIID_numc50_alpha0.1_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedsharplyavg-1000'

    "NonIID-alpha05"
    file_NonIID_05_pro = 'Cifar10_NonIID_numc50_alpha0.5_seed0-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_05_pro2 = 'Cifar10_NonIID_numc50_alpha0.5_seed0-K100-Dir-1.0-Res20-fedproavg2-1000'
    file_NonIID_05_avg = 'Cifar10_NonIID_numc50_alpha0.5_seed0-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_05_fa = 'Cifar10_NonIID_numc50_alpha0.5_seed0-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_05_fv = 'Cifar10_NonIID_numc50_alpha0.5_seed0-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_05_prox = 'Cifar10_NonIID_numc50_alpha0.5_seed0-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_05_sv = 'Cifar10_NonIID_numc50_alpha0.5_seed0-K100-Dir-1.0-Res20-fedsharplyavg-1000'

    "NonIID-alpha0.5-gaussian0-addmod0.4-dualityFalse"
    file_NonIID_05_0_04_F_pro = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_05_0_04_F_pro2 = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedproavg2-1000'
    file_NonIID_05_0_04_F_avg = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_05_0_04_F_fa = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_05_0_04_F_fv = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_05_0_04_F_prox = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_05_0_04_F_sv = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedsharplyavg-1000'

    "NonIID-alpha0.5-gaussian0.4-addmod0-dualityFalse"
    file_NonIID_05_04_0_F_pro = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_05_04_0_F_avg = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_05_04_0_F_fa = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_05_04_0_F_fv = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_05_04_0_F_prox = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_05_04_0_F_sv = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedsharplyavg-1000'

    "NonIID-alpha0.5-gaussian0.2-addmod0.2-dualityFalse"
    file_NonIID_05_02_02_F_pro = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_05_02_02_F_pro2 = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedproavg2-1000'
    file_NonIID_05_02_02_F_avg = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_05_02_02_F_fa = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_05_02_02_F_fv = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_05_02_02_F_prox = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_05_02_02_F_sv = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedsharplyavg-1000'

    "NonIID-alpha0.5-gaussian0.4-addmod0.4-dualityTrue"
    file_NonIID_05_04_04_T_pro = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_05_04_04_T_pro2 = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedproavg2-1000'
    file_NonIID_05_04_04_T_avg = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_05_04_04_T_fa = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_05_04_04_T_fv = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_05_04_04_T_prox = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_05_04_04_T_sv = 'Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedsharplyavg-1000'

    "NonIID-alpha1"
    file_NonIID_1_pro = 'Cifar10_NonIID_numc50_alpha1_seed0-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_1_avg = 'Cifar10_NonIID_numc50_alpha1_seed0-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_1_fa = 'Cifar10_NonIID_numc50_alpha1_seed0-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_1_fv = 'Cifar10_NonIID_numc50_alpha1_seed0-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_1_prox = 'Cifar10_NonIID_numc50_alpha1_seed0-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_1_svBig = 'Cifar10_NonIID_numc50_alpha1_seed0-K100-Dir-1.0-Res20-fedsharplyavg-1000-big'

    "NonIID-alpha1-gaussian0-addmod0.4-dualityFalse"
    file_NonIID_1_0_04_F_pro = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_1_0_04_F_avg = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_1_0_04_F_fa = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_1_0_04_F_fv = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_1_0_04_F_prox = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_1_0_04_F_svBig = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0_addmod0.4_dualityFalse-K100-Dir-1.0-Res20-fedsharplyavg-1000-big'

    "NonIID-alpha1-gaussian0.4-addmod0-dualityFalse"
    file_NonIID_1_04_0_F_pro = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_1_04_0_F_avg = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_1_04_0_F_fa = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_1_04_0_F_fv = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_1_04_0_F_prox = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_1_04_0_F_svBig = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.4_addmod0_dualityFalse-K100-Dir-1.0-Res20-fedsharplyavg-1000-big'

    "NonIID-alpha1-gaussian0.2-addmod0.2-dualityFalse"
    file_NonIID_1_02_02_F_pro = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_1_02_02_F_avg = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_1_02_02_F_fa = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_1_02_02_F_fv = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_1_02_02_F_prox = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_1_02_02_F_svBig = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.2_addmod0.2_dualityFalse-K100-Dir-1.0-Res20-fedsharplyavg-1000-big'

    "NonIID-alpha1-gaussian0.4-addmod0.4-dualityTrue"
    file_NonIID_1_04_04_T_pro = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_1_04_04_T_avg = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_1_04_04_T_fa = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_1_04_04_T_fv = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_1_04_04_T_prox = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_1_04_04_T_svBig = 'Cifar10_NonIID_numc50_alpha1_seed0_gaussian0.4_addmod0.4_dualityTrue-K100-Dir-1.0-Res20-fedsharplyavg-1000-big'

    "NonIID-alpha10"
    file_NonIID_10_pro = 'Cifar10_NonIID_numc50_alpha10_seed0-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_10_avg = 'Cifar10_NonIID_numc50_alpha10_seed0-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_10_fa = 'Cifar10_NonIID_numc50_alpha10_seed0-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_10_fv = 'Cifar10_NonIID_numc50_alpha10_seed0-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_10_prox = 'Cifar10_NonIID_numc50_alpha10_seed0-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_10_sv = 'Cifar10_NonIID_numc50_alpha10_seed0-K100-Dir-1.0-Res20-fedsharplyavg-1000'

    "NonIID-alpha100"
    file_NonIID_100_pro = 'Cifar10_NonIID_numc50_alpha100_seed0-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_100_avg = 'Cifar10_NonIID_numc50_alpha100_seed0-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_100_fa = 'Cifar10_NonIID_numc50_alpha100_seed0-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_100_fv = 'Cifar10_NonIID_numc50_alpha100_seed0-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_100_prox = 'Cifar10_NonIID_numc50_alpha100_seed0-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_100_sv = 'Cifar10_NonIID_numc50_alpha100_seed0-K100-Dir-1.0-Res20-fedsharplyavg-1000'

    "NonIID-alpha1000000"
    file_NonIID_1000000_pro = 'Cifar10_NonIID_numc50_alpha1000000_seed0-K100-Dir-1.0-Res20-fedproavg-1000'
    file_NonIID_1000000_avg = 'Cifar10_NonIID_numc50_alpha1000000_seed0-K100-Dir-1.0-Res20-fedavg-1000'
    file_NonIID_1000000_fa = 'Cifar10_NonIID_numc50_alpha1000000_seed0-K100-Dir-1.0-Res20-fedfa-1000'
    file_NonIID_1000000_fv = 'Cifar10_NonIID_numc50_alpha1000000_seed0-K100-Dir-1.0-Res20-fedfv-1000'
    file_NonIID_1000000_prox = 'Cifar10_NonIID_numc50_alpha1000000_seed0-K100-Dir-1.0-Res20-fedprox-1000'
    file_NonIID_1000000_sv = 'Cifar10_NonIID_numc50_alpha1000000_seed0-K100-Dir-1.0-Res20-fedsharplyavg-1000'


    'IID-alpha0'
    getLogGlobalPicture(file_IID_0_avg, file_IID_0_fv, file_IID_0_fa, file_IID_0_svBig, file_IID_0_pro, file_IID_0_prox, Type='alg')
    getLogGlobalPicture(file_IID_0_0_04_F_avg, file_IID_0_0_04_F_fa, file_IID_0_0_04_F_fv, file_IID_0_0_04_F_svBig, file_IID_0_0_04_F_pro, file_IID_0_0_04_F_prox, Type='alg')
    getLogGlobalPicture(file_IID_0_04_0_F_avg , file_IID_0_04_0_F_fa, file_IID_0_04_0_F_fv, file_IID_0_04_0_F_svBig, file_IID_0_04_0_F_pro, file_IID_0_04_0_F_prox, Type='alg')
    getLogGlobalPicture(file_IID_0_02_02_F_avg, file_IID_0_02_02_F_fa, file_IID_0_02_02_F_fv, file_IID_0_02_02_F_svBig, file_IID_0_02_02_F_pro, file_IID_0_02_02_F_prox, Type='alg')
    getLogGlobalPicture(file_IID_0_04_04_T_avg, file_IID_0_04_04_T_fa, file_IID_0_04_04_T_fv, file_IID_0_04_04_T_svBig, file_IID_0_04_04_T_pro, file_IID_0_04_04_T_prox, Type='alg')

    'NonIID-alpha0.1'
    # getLogGlobalPicture(file_NonIID_01_pro , file_NonIID_01_avg , file_NonIID_01_fa , file_NonIID_01_fv, file_NonIID_01_sv, Type= 'alg')
    # getLogGlobalPicture(file_NonIID_01_0_04_F_pro, file_NonIID_01_0_04_F_avg, file_NonIID_01_0_04_F_fa, file_NonIID_01_0_04_F_fv, file_NonIID_01_0_04_F_sv, Type='alg')
    # getLogGlobalPicture(file_NonIID_01_04_0_F_pro, file_NonIID_01_04_0_F_avg , file_NonIID_01_04_0_F_fa, file_NonIID_01_04_0_F_fv, file_NonIID_01_04_0_F_sv, Type='alg')
    # getLogGlobalPicture(file_NonIID_01_02_02_F_pro, file_NonIID_01_02_02_F_avg, file_NonIID_01_02_02_F_fa, file_NonIID_01_02_02_F_fv, file_NonIID_01_02_02_F_sv, Type='alg')
    # getLogGlobalPicture(file_NonIID_01_04_04_T_pro, file_NonIID_01_04_04_T_avg, file_NonIID_01_04_04_T_fa, file_NonIID_01_04_04_T_fv, file_NonIID_01_04_04_T_sv, Type='alg')

    'NonIID-alpha0.5'
    # getLogGlobalPicture(file_NonIID_05_avg , file_NonIID_05_fa , file_NonIID_05_fv, file_NonIID_05_pro, file_NonIID_05_sv, Type= 'alg')   #重跑pro
    # getLogGlobalPicture(file_NonIID_05_0_04_F_avg, file_NonIID_05_0_04_F_fa, file_NonIID_05_0_04_F_fv, file_NonIID_05_0_04_F_pro, file_NonIID_05_0_04_F_sv, Type='alg')
    # getLogGlobalPicture(file_NonIID_05_04_0_F_avg , file_NonIID_05_04_0_F_fa, file_NonIID_05_04_0_F_fv, file_NonIID_05_04_0_F_pro, file_NonIID_05_04_0_F_sv, Type='alg')
    # getLogGlobalPicture(file_NonIID_05_02_02_F_avg, file_NonIID_05_02_02_F_fa, file_NonIID_05_02_02_F_fv, file_NonIID_05_02_02_F_pro, file_NonIID_05_02_02_F_pro2,file_NonIID_05_02_02_F_sv, Type='alg')
    # getLogGlobalPicture(file_NonIID_05_04_04_T_avg, file_NonIID_05_04_04_T_fa, file_NonIID_05_04_04_T_fv, file_NonIID_05_04_04_T_pro, file_NonIID_05_04_04_T_sv, Type='alg')

    'NonIID-alpha1'
    # getLogGlobalPicture(file_NonIID_1_avg, file_NonIID_1_fa , file_NonIID_1_fv, file_NonIID_1_svBig, file_NonIID_1_pro, Type= 'alg')                   #缺pro
    # getLogGlobalPicture(file_NonIID_1_0_04_F_avg, file_NonIID_1_0_04_F_fa, file_NonIID_1_0_04_F_fv, file_NonIID_1_0_04_F_svBig, file_NonIID_1_0_04_F_pro, Type='alg')       #缺pro
    # getLogGlobalPicture(file_NonIID_1_04_0_F_avg, file_NonIID_1_04_0_F_fa, file_NonIID_1_04_0_F_fv, file_NonIID_1_04_0_F_svBig, file_NonIID_1_04_0_F_pro, Type='alg')       #缺pro
    # getLogGlobalPicture(file_NonIID_1_02_02_F_avg, file_NonIID_1_02_02_F_fa, file_NonIID_1_02_02_F_fv, file_NonIID_1_02_02_F_svBig, file_NonIID_1_02_02_F_pro, Type='alg')       #缺pro
    # getLogGlobalPicture(file_NonIID_1_04_04_T_avg, file_NonIID_1_04_04_T_fa, file_NonIID_1_04_04_T_fv, file_NonIID_1_04_04_T_svBig, file_NonIID_1_04_04_T_pro, Type='alg')       #缺pro

    'NonIID-alpha10'
    # getLogGlobalPicture(file_NonIID_10_avg, file_NonIID_10_fa, file_NonIID_10_fv, file_NonIID_10_sv, Type='alg')       #缺pro

    'NonIID-alpha100'
    # getLogGlobalPicture(file_NonIID_100_pro, file_NonIID_100_avg, file_NonIID_100_fa, file_NonIID_100_fv, file_NonIID_100_sv, Type='alg')

    'NonIID-alpha1000000'
    # getLogGlobalPicture(file_NonIID_1000000_pro, file_NonIID_1000000_avg, file_NonIID_1000000_fa, file_NonIID_1000000_fv, file_NonIID_1000000_sv, Type='alg')





