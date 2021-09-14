import time
import torch
from pykeops.torch import LazyTensor
from torchvision import datasets, transforms, models
import numpy as np
from inclearn import parser
from inclearn.lib import factory
import torch.nn as nn
from inclearn.lib.data.incdataset import DummyDataset
from torch.utils.data import DataLoader


use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64

def KMeans(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c

def KMeans_cosine(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Cosine similarity metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids
    # Normalize the centroids for the cosine similarity:
    c = torch.nn.functional.normalize(c, dim=1, p=2)

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        S_ij = x_i | c_j  # (N, K) symbolic Gram matrix of dot products
        cl = S_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Normalize the centroids, in place:
        c[:] = torch.nn.functional.normalize(c, dim=1, p=2)

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the cosine similarity with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c

def find_cluster(y, cl, pos_mask, K, size=10):
    for z in range(100):
        thd = 100 - z
        clusters = []
        for i in range(K):
            clusters.append([])
        for cat in range(100):
            if not pos_mask[cat]:
                continue
            pos = y == cat
            y_cl = cl[pos]
            max_cat = -1
            max_ratio = -1
            num = y_cl.shape[0]
            for i in range(K):
                s = (y_cl == i).sum()
                ratio = s / num * 100
                if ratio > max_ratio:
                    max_ratio = ratio
                    max_cat = i

            if max_ratio > thd:
                clusters[max_cat].append(cat)
        L_max = -1
        id_max = -1
        for i in range(K):
            if len(clusters[i]) > L_max:
                L_max = len(clusters[i])
                id_max = i
        if L_max >= size:
            return clusters[id_max][:10], thd



if __name__ == '__main__':
    args = parser.get_parser().parse_args()
    args = vars(args)
    factory.set_device(args)
    device = args["device"][0]

    resnet18 = models.resnet18(pretrained=True)
    #resnet18 = nn.Sequential(*list(resnet18.children())[:-4]).to(device)
    resnet18 = nn.Sequential(*list(resnet18.children())[:-2]).to(device)
    pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)

    data_path = "../CIL-backbone/DER-ClassIL.pytorch/data/cifar100/"
    base_dataset = datasets.cifar.CIFAR100
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    train_dataset = base_dataset(data_path, train=True, download=True)
    test_dataset = base_dataset(data_path, train=False, download=True)
    x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
    x_test, y_test = test_dataset.data, np.array(test_dataset.targets)

    x_all = np.concatenate((x_train, x_test))
    y_all = np.concatenate((y_train, y_test))

    trsf = transforms.Compose([*common_transforms])

    loader = DataLoader(
        DummyDataset(x_all, y_all, y_all, trsf),
        batch_size=128,
        shuffle=False,
        num_workers=8,
        batch_sampler=None
    )

    feats = []
    with torch.no_grad():
        for data in loader:
            x = data["inputs"].to(device)
            y = data["targets"].to(device)
            B = x.shape[0]
            feat = pool(resnet18(x)).view(B, -1)
            feats.append(feat)
    feats = torch.cat(feats, dim=0)

    pos_mask = torch.ones((100, ), dtype=bool)
    feats_mask = torch.ones((60000, ), dtype=bool)
    cluster_res = []
    thd_res = []

    for i in range(9):
        for j in range(100):
            if not pos_mask[j]:
                feats_mask[(y_all == j)] = False
        K = 10 - i
        cl, c = KMeans(x=feats[feats_mask], K=K, Niter=100)
        cluster, thd = find_cluster(y_all[feats_mask], cl, pos_mask, K, size=10)
        cluster_res.append(cluster)
        thd_res.append(thd)
        for y in cluster:
            pos_mask[y] = False

    cluster = []
    for cat in range(100):
        if pos_mask[cat]:
            cluster.append(cat)
    cluster_res.append(cluster)
    thd_res.append(thd)
    order = []
    for i in range(10):
        print("cluster {}, thd {}".format(cluster_res[i], thd_res[i]))
        order += cluster_res[i]
    print(order)