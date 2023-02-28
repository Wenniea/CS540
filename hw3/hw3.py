from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    x = np.load(filename)
    return x - np.mean(x, axis=0)


def get_covariance(dataset):
    x = dataset
    transposeX = np.transpose(x)
    S = np.dot(transposeX, x) / (len(x) - 1)
    return S


def get_eig(S, m):
    eigVal, eigVec = eigh(S, subset_by_index=(len(S) - m, len(S) - 1))
    return np.diag(eigVal[::-1]), eigVec[:, ::-1]


def get_eig_prop(S, prop):
    eigVal, eigVec = eigh(S, subset_by_value=[prop * np.sum(np.linalg.eigvals(S)), np.inf])
    return np.diag(eigVal[::-1]), eigVec[:, ::-1]


def project_image(image, U):
    transposeU = np.transpose(U)
    projection = np.dot(U, transposeU)
    return np.dot(projection, image)


def display_image(orig, proj):
    orig_image = np.transpose(np.reshape(orig, (32, 32)))
    proj_image = np.transpose(np.reshape(proj, (32, 32)))
    fig, (figL, figR) = plt.subplots(1, 2)
    figL.set_title('Original')
    figR.set_title('Projection')

    Lval = figL.imshow(orig_image, aspect='equal')
    Rval = figR.imshow(proj_image, aspect='equal')

    fig.colorbar(Lval, ax=figL)
    fig.colorbar(Rval, ax=figR)
    plt.show()

