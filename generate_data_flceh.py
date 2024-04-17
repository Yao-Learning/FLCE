# time: 20230822
# author: guo qi
# description: generate data for federated learning
import numpy as np
import random
from torchvision import datasets, transforms

def balance_split(num_clients, num_samples):
    """Assign same sample sample for each client.

    Args:
        num_clients (int): Number of clients for partition.
        num_samples (int): Total number of samples.

    Returns:
        numpy.ndarray: A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.

    """
    num_samples_per_client = int(num_samples / num_clients)
    client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(
        int)
    diff = np.sum(client_sample_nums) - num_samples  # diff <= 0
    # Add/Subtract the excess number starting from first client
    if diff != 0:
        for cid in range(num_clients):
            if client_sample_nums[cid] > diff:
                client_sample_nums[cid] -= diff
                break
    return client_sample_nums


def split_indices(num_cumsum, rand_perm):
    '''
    :param num_cumsum:
    :param rand_perm:
    :return: client_dict
    '''
    client_indices_pairs = [(cid, idxs) for cid, idxs in
                            enumerate(np.split(rand_perm, num_cumsum)[:-1])]
    client_dict = dict(client_indices_pairs)
    return client_dict


def homo_partition(client_sample_nums, num_samples):
    """Partition data indices in IID way given sample numbers for each clients.

    Args:
        client_sample_nums (numpy.ndarray): Sample numbers for each clients.
        num_samples (int): Number of samples.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    rand_perm = np.random.permutation(num_samples)
    num_cumsum = np.cumsum(client_sample_nums).astype(int)
    client_dict = split_indices(num_cumsum, rand_perm)
    return client_dict


def hetero_dir_partition(targets, num_clients, num_classes, dir_alpha, min_require_size=None):
    """

    Non-iid partition based on Dirichlet distribution. The method is from "hetero-dir" partition of
    `Bayesian Nonparametric Federated Learning of Neural Networks <https://arxiv.org/abs/1905.12022>`_
    and `Federated Learning with Matched Averaging <https://arxiv.org/abs/2002.06440>`_.

    This method simulates heterogeneous partition for which number of data points and class
    proportions are unbalanced. Samples will be partitioned into :math:`J` clients by sampling
    :math:`p_k \sim \text{Dir}_{J}(\alpha)` and allocating a :math:`p_{p,j}` proportion of the
    samples of class :math:`k` to local client :math:`j`.

    Sample number for each client is decided in this function.

    Args:
        targets (list or numpy.ndarray): Sample targets. Unshuffled preferred.
        num_clients (int): Number of clients for partition.
        num_classes (int): Number of classes in samples.
        dir_alpha (float): Parameter alpha for Dirichlet distribution.
        min_require_size (int, optional): Minimum required sample number for each client. If set to ``None``, then equals to ``num_classes``.

    Returns:
        dict: ``{ client_id: indices}``.
    """
    if min_require_size is None:
        min_require_size = num_classes

    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    num_samples = targets.shape[0]

    min_size = 0
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        # for each class in the dataset
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(
                np.repeat(dir_alpha, num_clients))
            # Balance
            proportions = np.array(
                [p * (len(idx_j) < num_samples / num_clients) for p, idx_j in
                 zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                         zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    client_dict = dict()
    for cid in range(num_clients):
        np.random.shuffle(idx_batch[cid])
        client_dict[cid] = np.array(idx_batch[cid])

    return client_dict


def generate_index(train_data, train_targets, num_clients, num_classes, min_require_size, tag_distribution, dir_alpha, seed):
    '''
    :param train_data:
    :param train_targets:
    :param num_clients:
    :param num_classes:
    :param min_require_size:
    :param tag_distribution:
    :param dir_alpha:
    :param seed:
    :return: client_dict: {client_id: indices}
    '''
    # Setting the random seed for reproducibility
    np.random.seed(seed)

    num_samples = len(train_targets)
    # tag_distribution = "NonIID"  # IID or NonIID

    if tag_distribution == "IID":
        # generate IID data
        client_sample_nums = balance_split(num_clients, num_samples)
        client_dict = homo_partition(client_sample_nums, num_samples)
        # print(len(client_dict))
    elif tag_distribution == "NonIID":
        # generate NonIID data
        targets = np.array(train_targets)
        targets = targets.flatten()
        client_dict = hetero_dir_partition(targets, num_clients, num_classes, dir_alpha, min_require_size)
        # print(len(client_dict))
    else:
        client_dict = None
        print("error. There is no IID or NonIID.")
    # print(len(client_dict))
    return client_dict


import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10, FashionMNIST, MNIST
import os

def load_dataset(dataset_name: str):

    path_dataset = "/data/yaominghao/data/newdata/"
    dataset_name = dataset_name.upper()

    if dataset_name.upper() == 'FASHIONMNIST':
        # famnist_train = torchvision.datasets.FashionMNIST(root='/data/yaominghao/data/newdata/', train=True, download=True,
        #                                                 transform=transforms.ToTensor())
        # famnist_test = torchvision.datasets.FashionMNIST(root='/data/yaominghao/data/newdata/', train=False, download=True,
        #                                                transform=transforms.ToTensor())
        # 定义数据变换
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        # 加载训练集
        train_dataset = datasets.FashionMNIST(root='/data/yaominghao/data/newdata/', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)

        # 取出训练数据和训练标签
        train_data, train_targets = next(iter(train_loader))
        train_data = train_data.numpy()
        train_targets = train_targets.numpy().tolist()
        # train_data = train_data.view(len(train_data), -1)  # 将图片数据展平

        # 加载测试集
        test_dataset = datasets.FashionMNIST(root='/data/yaominghao/data/newdata/', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

        # 取出测试数据和测试标签
        test_data, test_targets = next(iter(test_loader))
        test_data = test_data.numpy()
        test_targets = test_targets.numpy().tolist()
        # test_data = test_data.view(len(test_data), -1)  # 将图片数据展平

        # dset_train = getattr(torchvision.datasets, 'FashionMNIST')
        # train_data, train_targets = dset_train.data, dset_train.targets
        #
        # dset_test = getattr(torchvision.datasets, 'FashionMNIST')
        # test_data, test_targets = dset_test.data, dset_test.targets
    else:
        dset_train = getattr(torchvision.datasets, dataset_name.upper())
        dset_test = getattr(torchvision.datasets, dataset_name.upper())

    if 'CIFAR' in dataset_name.upper():
        dset_train = dset_train(path_dataset, train=True, download=True)
        train_data, train_targets = dset_train.data, dset_train.targets

        dset_test = dset_test(path_dataset, train=False, download=True)
        test_data, test_targets = dset_test.data, dset_test.targets

    return train_data, train_targets, test_data, test_targets

def generate_data_fl(save_path_input):
    # Setting the random seed for reproducibility
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset_loadname = "FASHIONMNIST"
    # dataset_loadname = "cifar10"
    # step1: Load dataset
    train_data, train_targets, test_data, test_targets = load_dataset(dataset_loadname)

    # step2: Design data allocation scheme for all participants
    num_clients = 50
    num_classes = 10
    min_require_size = 100
    tag_distribution = "IID"  # IID or NonIID
    dir_alpha = 0
    client_dict = generate_index(train_data, train_targets, num_clients, num_classes, min_require_size, tag_distribution, dir_alpha, seed)

    # step3: Distribute data for each participant by dataset and scheme
    train_data, train_targets = np.array(train_data), np.array(train_targets)
    test_data, test_targets = np.array(test_data), np.array(test_targets)
    data_total = {}
    for cid in range(num_clients):
        data_total[cid] = {}
        data_total[cid]['train_data'] = train_data[client_dict[cid]]
        data_total[cid]['train_targets'] = train_targets[client_dict[cid]]

    # Distribute test data at last virtual client
    data_total[num_clients] = {}
    data_total[num_clients]['test_data'] = test_data
    data_total[num_clients]['test_targets'] = test_targets

    save_path = save_path_input
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = dataset_loadname + "_" + tag_distribution + "_numc" + str(num_clients) \
               + "_alpha" + str(dir_alpha) + "_seed" + str(seed) + ".npy"
    save_name = save_path + filename
    np.save(save_name, data_total)

    return data_total, filename

def load_data_total(data_dir, name_data_total):
    '''

    :param data_dir: pathname of load data
    :param name_data_total: filename of load data
    :return: data_total is a dict, the last element is test data and targets, other element is the data of each client.
    '''
    # data_dir = "../data/data_generate_result/"
    # name_data_total = "data_total_svhn_NonIID_alpha0.3_num60_pro123.npy"
    # data_dir_specific = os.path.join(data_dir, name_data_total)

    data_dir_specific = os.path.join(data_dir, name_data_total)
    data_total_temp = np.load(data_dir_specific, allow_pickle=True)

    data_total = data_total_temp.item()

    return data_total


def add_gaussian_noise_based_on_data(data):
    """Add Gaussian noise to the data based on its mean and standard deviation."""
    mean = 0  # np.mean(data)
    std = np.std(data)
    noise = np.random.normal(mean, std, data.shape)
    noisy_data = data + noise
    # clipping
    # noisy_data = np.clip(noisy_data, 0, 255).astype(np.uint8)
    # normalization
    noisy_data = (noisy_data - np.min(noisy_data)) / (np.max(noisy_data) - np.min(noisy_data))
    noisy_data = (noisy_data * 255).astype(np.uint8)

    return noisy_data


def label_mapping(labels, mapping_type):
    """Map labels according to the given mapping type."""
    # Simple example: Add 1 and mod 10
    if mapping_type == 'addmod':
        return (labels + 1) % 10
    # TODO: Handle other mapping types here
    return labels


def modify_data(data_total, noise_type, noise_rate, mapping_type, mapping_rate, duality):
    """Modify the data_total dataset based on noise and mapping specifications."""
    client_ids = list(data_total.keys())
    client_ids = client_ids[:-1]  # Remove the last client (test data)

    num_clients_noise = int(len(client_ids) * noise_rate)
    num_clients_mapping = int(len(client_ids) * mapping_rate)

    # # Randomly select clients for noise addition
    # clients_for_noise = random.sample(client_ids, num_clients_noise)
    # # Randomly select clients for label mapping
    # clients_for_mapping = random.sample(client_ids, num_clients_mapping)

    if duality is True:
        clients_for_noise = [cid for cid in client_ids if cid >= (len(client_ids) - num_clients_noise)]
        clients_for_mapping = [cid for cid in client_ids if cid >= (len(client_ids) - num_clients_mapping)]
    else:
        clients_for_noise = [cid for cid in client_ids if cid >= (len(client_ids) - num_clients_noise)]
        clients_for_mapping = [cid for cid in client_ids if
                               (len(client_ids) - num_clients_noise - num_clients_mapping) <= cid < (
                                           len(client_ids) - num_clients_noise)]

    # Modify data based on noise type and mapping type
    for client_id in client_ids:
        if client_id in clients_for_noise:
            if noise_type == 'gaussian':
                data_total[client_id]['train_data'] = add_gaussian_noise_based_on_data(
                    data_total[client_id]['train_data'])
            # TODO: Handle other noise types here

        if client_id in clients_for_mapping:
            data_total[client_id]['train_targets'] = label_mapping(data_total[client_id]['train_targets'], mapping_type)
            # TODO: Handle other mapping types here

    return data_total


def transform_data_quality(data_dir, name_data_total, noise_type, noise_rate, mapping_type, mapping_rate, duality):
    ''' Transform a standard dataset into a new dataset with noise and label mapping.
    :param data_total: a dict, the last element is test data and targets, other element is the data of each client.
    :param noise_type:
    :param noise_rate:
    :param mapping_type:
    :param mapping_rate:
    :return: data_total
    '''
    data_total = load_data_total(data_dir, name_data_total)
    data_total = modify_data(data_total, noise_type, noise_rate, mapping_type, mapping_rate, duality)
    # save data_total
    save_path = data_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = save_path + name_data_total[:-4] + "_" + noise_type + str(noise_rate) \
                + "_" + mapping_type + str(mapping_rate) + "_duality" + str(duality) + name_data_total[-4:]
    np.save(save_name, data_total)

if __name__ == '__main__':

    data_dir = "/data/yaominghao/data/newdata/FashionMINIST/"
    # name_data_total = "Cifar10_IID_numc50_alpha0_seed0.npy"
    noise_type = 'gaussian'
    noise_rate = 0
    mapping_type = 'addmod'
    mapping_rate = 0
    duality = False

    save_path_input = data_dir

    # generate data and save data
    _, filename = generate_data_fl(save_path_input)
    name_data_total = filename

    # load data from disk
    data_total = load_data_total(data_dir, name_data_total)
    transform_data_quality(data_dir, name_data_total, noise_type, noise_rate, mapping_type, mapping_rate, duality)

    print(data_total.keys())
    print(data_total[0].keys())
    print(data_total[0]['train_data'].shape)
    print(data_total[0]['train_targets'].shape)
    print(data_total[50].keys())
    print(data_total[50]['test_data'].shape)
    print(data_total[50]['test_targets'].shape)





