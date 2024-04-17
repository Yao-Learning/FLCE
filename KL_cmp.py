import numpy as np
from utils import completion_matrix
from scipy.special import kl_div
import torch
from scipy.stats import entropy

def getMatrixData(filename, Type):
        fileload = '/data/yaominghao/code/FedRepo/result/'
        if Type == 'raw':
                filename = fileload + filename + '.npy'
        elif Type == 'final':
                filename = fileload + 'final/' + filename + '_final.npy'
        else:
                print('Type ERROR!')

        data_matrix = np.load(filename, allow_pickle=True)
        return data_matrix

def getDataQuality(klType, dataType):
        data_matrix = []
        if klType == 'feature':
                if dataType == 'G0A0':
                        data_matrix = torch.ones(10, 50)
                elif dataType == 'G40A0':
                        data_matrix =  torch.cat((torch.ones(30), 0.32 * torch.ones(20))).repeat(10).reshape(10,-1)
                elif dataType == 'G0A40':
                        data_matrix = torch.cat((torch.ones(30), 0.000001 * torch.ones(20))).repeat(10).reshape(10, -1)
                elif dataType == 'G20A20':
                        data_matrix = torch.cat((torch.ones(30), 0.32 * torch.ones(10), 0.000001 * torch.ones(10))).repeat(10).reshape(10,-1)
                elif dataType == 'G40A40':
                        data_matrix = torch.cat((torch.ones(30), 0.000001 * torch.ones(20))).repeat(10).reshape(10, -1)
                else:
                        print('Type Error!')
        elif klType == 'client':
                if dataType == 'G0A0':
                        data_matrix = torch.ones(50, 10)
                elif dataType == 'G40A0':
                        data_matrix =  torch.cat((torch.ones(30), 0.32 * torch.ones(20))).repeat(10).reshape(10,-1)
                elif dataType == 'G0A40':
                        data_matrix = torch.cat((torch.ones(30), 0.000001 * torch.ones(20))).repeat(10).reshape(10, -1)
                elif dataType == 'G20A20':
                        t1 = torch.ones(10).repeat(30)
                        t2 = 0.32 * torch.ones(10).repeat(10)
                        t3 = 0.01 * torch.ones(10).repeat(10)
                        data_matrix = torch.cat((t1, t2, t3)).reshape(50,-1)
                elif dataType == 'G40A40':
                        data_matrix = torch.cat((torch.ones(30), 0.000001 * torch.ones(20))).repeat(10).reshape(10, -1)
                else:
                        print('Type Error!')
        elif klType == 'alg':
                if dataType == 'G0A0':
                        data_matrix = torch.ones(50)
                elif dataType == 'G40A0':
                        data_matrix =  torch.cat((torch.ones(30), 0.32 * torch.ones(20)))
                elif dataType == 'G0A40':
                        data_matrix = torch.cat((torch.ones(30), 0.000001 * torch.ones(20)))
                elif dataType == 'G20A20':
                        t1 = torch.ones(30)
                        t2 = 0.32 * torch.ones(10)
                        t3 = 0.000001 * torch.ones(10)
                        data_matrix = torch.cat((t1, t2, t3))
                elif dataType == 'G40A40':
                        data_matrix = torch.cat((torch.ones(30), 0.000001 * torch.ones(20)))
                else:
                        print('Type Error!')
        else:
                print('Type Error!')


        return data_matrix

def KL_divergence(list1, list2):
        KL_feature_list = []
        for i in range(len(list1)):
                tmp1 = list1[i]
                tmp2 = list2[i].numpy()
                tmp3 = tmp2/tmp1
                tmp4 = np.log(tmp3)
                # kl_divergence = np.sum(tmp1 * tmp4)
                # kl_divergence = entropy(tmp2, tmp1)

                KL_feature_list.append(kl_divergence)
        return KL_feature_list

def KLnorm(list):
        min_val = min(list)
        max_val = max(list)
        normalized_data = [(x - min_val) / (max_val - min_val) * 0.99 + 0.01 for x in list]
        return normalized_data


if __name__ == '__main__':
        # filename = 'contribution_matrixCifar10_IID_numc50_alpha0_seed0_gaussian0.2_addmod0.2_dualityFalse'
        # filename = 'contribution_matrixCifar10_NonIID_numc50_alpha1000000_seed0'
        filename = 'contribution_matrixCifar10_IID_numc50_alpha0_seed0'
        data = getMatrixData(filename, Type='raw')
        Matrix_data = np.sum(getMatrixData(filename, Type='raw'), axis=0)
        Matrix_data_2 = np.sum(Matrix_data, axis=0)
        # final_Matrix_data = np.sum(getMatrixData(filename, Type='final'), axis=0)
        # print(Matrix_data)

        # print(final_Matrix_data)
        DataQuality = getDataQuality(klType='alg', dataType='G0A0')

        kl_divergence = entropy(KLnorm(Matrix_data_2), DataQuality)

        fedavg_matrix = list(1 for _ in range(50))

        kl_avg = entropy(fedavg_matrix, DataQuality)

        print(kl_divergence)

        print(kl_avg)

        # KL = KL_divergence(Matrix_data, DataQuality)
        # print(kl_divergence)

