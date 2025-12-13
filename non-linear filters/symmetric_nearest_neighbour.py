import numpy as np

img = np.array([
        [1, 4, 5, 6, 5, 8, 4, 6, 1, 1, 3, 1],
        [1, 4, 2, 4, 5, 3, 8, 9, 1, 5, 1, 5],
        [0, 3, 4, 5, 7, 2, 8, 6, 1, 3, 6, 1],
        [1, 3, 4, 3, 7, 1, 7, 6, 1, 1, 5, 1],
        [1, 4, 5, 6, 5, 8, 4, 6, 3, 1, 3, 3],
        [1, 2, 3, 3, 5, 6, 7, 8, 7, 6, 5, 4]
])

def get_neighborhood(img, r, c):
    return img[r-1:r+2, c-1:c+2].flatten()

def symmetric_nearest_neighbor(neighborhood):
    center = neighborhood[4]  # center of 3×3 flattened
    pairs = [(0,8), (1,7), (2,6), (3,5)]
    selected = [center]

    for i, j in pairs:
        if abs(neighborhood[i] - center) <= abs(neighborhood[j] - center):
            selected.append(neighborhood[i])
        else:
            selected.append(neighborhood[j])

    return np.mean(selected)


if __name__ == "__main__":
    targets = [(2,3), (2,4), (2,5)]  # zero-based (row, col)

    for idx, (r, c) in zip([28, 29, 30], targets):
        n = get_neighborhood(img, r, c)
        print(f"Index {idx}:")
        print("  SNN:", symmetric_nearest_neighbor(n))