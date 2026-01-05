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

def erosion(image, se, anchor):
    se, anchor = flip_se(se, anchor)
    img_h, img_w = image.shape
    se_h, se_w = se.shape
    ax, ay = anchor
    output = np.zeros_like(image)

    for i in range(img_h):
        for j in range(img_w):
            ok = True
            for u in range(se_h):
                for v in range(se_w):
                    if se[u, v] != 1:
                        continue
                    ii = i + (u - ax)
                    jj = j + (v - ay)
                    if ii < 0 or ii >= img_h or jj < 0 or jj >= img_w:
                        ok = False
                        break
                    if image[ii, jj] != 1:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                output[i, j] = 1
    return output


def dilation(image, se, anchor):
    se, anchor = flip_se(se, anchor)
    img_h, img_w = image.shape
    se_h, se_w = se.shape
    ax, ay = anchor
    output = np.zeros_like(image)

    for i in range(img_h):
        for j in range(img_w):
            hit = False
            for u in range(se_h):
                for v in range(se_w):
                    if se[u, v] != 1:
                        continue
                    ii = i + (u - ax)
                    jj = j + (v - ay)
                    if 0 <= ii < img_h and 0 <= jj < img_w:
                        if image[ii, jj] == 1:
                            hit = True
                            break
                if hit:
                    break
            if hit:
                output[i, j] = 1
    return output


def opening(image, se, anchor):
    return dilation(erosion(image, se, anchor), se, anchor)

# Example usage
image = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

se_a = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

anchor = (0, 1)

result_a = opening(image, se_a, anchor)

print(result_a)