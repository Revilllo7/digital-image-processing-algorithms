import numpy as np

def flip_se(se, anchor):
    # Ensure se is 2D
    if se.ndim == 1:
        se = se.reshape(1, -1)
    se_flipped = np.flipud(np.fliplr(se))
    h, w = se.shape
    ax, ay = anchor
    anchor_flipped = (h - 1 - ax, w - 1 - ay)
    return se_flipped, anchor_flipped


def hit_or_miss(image, se, anchor):
    """
    Perform Hit-or-Miss transform on a binary image.

    Parameters:
    - image: 2D numpy array (0/1)
    - se (structural element): 2D numpy array with values {1, 0, None}
    - anchor: (row, col) index of anchor in SE

    Returns:
    - output: 2D numpy array (0/1)
    """

    se, anchor = flip_se(se, anchor)

    img_h, img_w = image.shape
    se_h, se_w = se.shape
    ax, ay = anchor

    output = np.zeros_like(image)

    for i in range(img_h):
        for j in range(img_w):
            match = True

            for u in range(se_h):
                for v in range(se_w):

                    se_val = se[u, v]
                    if se_val is None:   # '*'
                        continue

                    ii = i + (u - ax)
                    jj = j + (v - ay)

                    # Outside image = 0
                    if ii < 0 or ii >= img_h or jj < 0 or jj >= img_w:
                        img_val = 0
                    else:
                        img_val = image[ii, jj]

                    if img_val != se_val:
                        match = False
                        break

                if not match:
                    break

            if match:
                output[i, j] = 1

    return output

# Example usage
image = np.array([
    [1,1,1,1,1,1,1],
    [1,0,1,1,1,0,1],
    [1,1,0,0,0,1,1],
    [0,1,0,1,0,1,0],
    [0,1,1,1,1,1,0]
])

# SE (a)
print("subpoint A:")
se_a = np.array([
    [1, None, 0],
    [None, 1, None],
    [0, None, 1]
], dtype=object)

anchor = (1, 1)

result_a = hit_or_miss(image, se_a, anchor)
print(result_a)

print()
# 

# SE (b)
print("subpoint B:")
se_b = np.array([
    [1, None, 0],
    [1, 0, None],
    [1, 1, 1]
], dtype=object)

anchor = (1, 1)

result_b = hit_or_miss(image, se_b, anchor)
print(result_b)

print()
# 

# SE (c)
print("subpoint C:")
se_c = np.array([
    [0, 0, 0],
    [None, 1, None],
    [1, 1, 1]
], dtype=object)

anchor = (1, 1)

result_c = hit_or_miss(image, se_c, anchor)
print(result_c)

print()
# 

# SE (d)
print("subpoint D:")
se_d = np.array([
    [None, 0, 0],
    [1, 1, 0],
    [None, 1, None]
], dtype=object)

anchor = (1, 1)

result_d = hit_or_miss(image, se_d, anchor)
print(result_d)