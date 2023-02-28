import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram



def load_data(filepath):
    data = []
    with open(filepath, mode='r') as f:
        file = csv.DictReader(f)
        for row in file:
            data.append(row)
    return data


def calc_features(row):
    stats = np.array([row['Attack'], row['Sp. Atk'], row['Speed'], row['Defense'], row['Sp. Def'],
                      row['HP']]).astype(int)
    return stats


def hac(features):
    distance_matrix = np.zeros(shape=(len(features), len(features)), dtype=float)
    r, c = np.tril_indices(len(features), k=-1)
    for i, j in zip(r, c):
        poke_one_squared = np.square(features[i])
        poke_two_squared = np.square(features[j])
        distance = np.sqrt(sum(poke_one_squared) + np.sum(poke_two_squared) - (2 * np.dot(
            features[i], np.transpose(features[j]))))
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

    available_cluster = list(range(len(distance_matrix)))
    cluster_size = [1] * len(distance_matrix)
    Z = np.zeros(shape=(len(features) - 1, 4), dtype=float)

    for i in range(len(Z)):
        distance_matrix[distance_matrix == 0] = np.inf
        indices = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)


        smallerIndex = available_cluster.index(min(indices))
        largerIndex = available_cluster.index(max(indices))

        smallDIndex = min(indices)
        largeDIndex = max(indices)

        available_cluster.pop(largerIndex)
        available_cluster.pop(smallerIndex)
        available_cluster.append(len(features) + i)

        Z[i, 0] = smallDIndex
        Z[i, 1] = largeDIndex
        Z[i, 2] = distance_matrix[indices[0], indices[1]]
        Z[i, 3] = cluster_size[smallDIndex] + cluster_size[largeDIndex]
        cluster_size.append(cluster_size[smallDIndex] + cluster_size[largeDIndex])

        new_row = np.zeros(shape=(1, len(features) + i), dtype=float)
        new_col = np.zeros(shape=(len(features) + i + 1, 1), dtype=float)
        distance_matrix = np.vstack([distance_matrix,new_row])
        distance_matrix = np.hstack([distance_matrix,new_col])
        for j in range(distance_matrix.shape[0]):
            if j == len(features) + i:
                distance_matrix[-1, j] = np.inf
            elif j == smallDIndex:
                distance_matrix[-1, j] = distance_matrix[largeDIndex, j]
                distance_matrix[j, -1] = distance_matrix[largeDIndex, j]
            elif j == largeDIndex:
                distance_matrix[-1, j] = distance_matrix[smallDIndex, j]
                distance_matrix[j, -1] = distance_matrix[smallDIndex, j]
            else:
                distance_matrix[-1, j] = max(distance_matrix[largeDIndex, j], distance_matrix[
                    smallDIndex, j])
                distance_matrix[j, -1] = max(distance_matrix[largeDIndex, j], distance_matrix[
                    smallDIndex,j])
        distance_matrix[smallDIndex, :] = float("inf")
        distance_matrix[largeDIndex, :] = float("inf")
        distance_matrix[:,smallDIndex] = float("inf")
        distance_matrix[:,largeDIndex] = float("inf")
    return Z


def imshow_hac(Z, names):
    plt.title("N = " + str(len(Z)+1))
    dendrogram(
        Z,
        leaf_rotation=90,
        labels=names
    )
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    features_and_names = [(calc_features(row), row['Name']) for row in
                          load_data('Pokemon.csv')[: 180]]
    Z = hac([row[0] for row in features_and_names])
    names = [row[1] for row in features_and_names]
    imshow_hac(Z, names)