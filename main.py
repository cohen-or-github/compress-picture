from PIL import Image
import numpy as np

def main():
    image = Image.open('avatar.jpg')
    matrix = np.array(image)
    red, green, blue = matrix[:, :, 0], matrix[:, :, 1], matrix[:, :, 2]
    print(red.shape)

    red_svd = np.linalg.svd(red, full_matrices=True, compute_uv=True)
    green_svd = np.linalg.svd(green, full_matrices=True, compute_uv=True)
    blue_svd = np.linalg.svd(blue, full_matrices=True, compute_uv=True)
    new_image = np.zeros(matrix.shape)

    k_values = [5, 100, 200, 250, 400]
    for k in k_values:
        new_image[:, :, 0] = low_rank_app(red_svd, k)
        new_image[:, :, 1] = low_rank_app(green_svd, k)
        new_image[:, :, 2] = low_rank_app(blue_svd, k)
        new = Image.fromarray(new_image.astype('uint8'))
        new.save(str(k) + " k -avatar.png")
        error = compute_error(red_svd[1], k)
        print("Error of red matrix for k = " + str(k) + " is: " + str(error))


def low_rank_app(svd, k):
    new_matrix = np.zeros((svd[0].shape[0], svd[2].shape[0]))
    for i in range(k):
        new_matrix[i][i] = svd[1][i]

    low_rank = np.matmul(new_matrix, svd[2])
    low_rank = np.matmul(svd[0], low_rank)

    return low_rank


def compute_error(singular_values, k):
    for i in singular_values:
        i = i ** 0.5

    k_sum = sum(singular_values[k+1:])
    total_sum = sum(singular_values[:])

    return k_sum / total_sum


if __name__ == '__main__':
    main()