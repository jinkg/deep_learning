import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


def zero_pad(X, pad):
    """
        Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
        as illustrated in Figure 1.

        Argument:
        X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
        pad -- integer, amount of padding around each image on vertical and horizontal dimensions

        Returns:
        X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
        """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)

    return X_pad


if __name__ == '__main__'"":
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = zero_pad(x, 2)
    print("x.shape =", x.shape)
    print("x_pad.shape =", x_pad.shape)
    print("x[1,1] =", x[1, 1])
    print("x_pad[1,1] =", x_pad[1, 1])

    fig, axarr = plt.subplots(1, 2)
    axarr[0].set_title('x')
    axarr[0].imshow(x[0, :, :, 0])
    axarr[1].set_title('x_pad')
    axarr[1].imshow(x_pad[0, :, :, 0])
    plt.show()


def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(W, a_slice_prev)
    Z = np.sum(s)
    Z += float(b)

    return Z


if __name__ == '__main__'"":
    np.random.seed(1)
    a_slice_prev = np.random.randn(4, 4, 3)
    W = np.random.randn(4, 4, 3)
    b = np.random.randn(1, 1, 1)

    Z = conv_single_step(a_slice_prev, W, b)
    print("Z =", Z)


def conv_forward(A_prev, W, b, hparameters):
    """
       Implements the forward propagation for a convolution function

       Arguments:
       A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
       W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
       b -- Biases, numpy array of shape (1, 1, 1, n_C)
       hparameters -- python dictionary containing "stride" and "pad"

       Returns:
       Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
       cache -- cache of values needed for the conv_backward() function
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    n_H = int((n_H_prev + 2 * pad - f) / stride + 1)
    n_W = int((n_W_prev + 2 * pad - f) / stride + 1)

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end]

                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

    assert (Z.shape == (m, n_H, n_W, n_C))

    cache = (A_prev, W, b, hparameters)

    return Z, cache


if __name__ == '__main__'"":
    np.random.seed(1)
    A_prev = np.random.randn(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad": 2,
                   "stride": 2}

    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    print("Z's mean =", np.mean(Z))
    print("Z[3,2,1] =", Z[3, 2, 1])
    print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])


def pool_forward(A_prev, hparameters, mode="max"):
    """
       Implements the forward pass of the pooling layer

       Arguments:
       A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
       hparameters -- python dictionary containing "f" and "stride"
       mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

       Returns:
       A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
       cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hparameters['f']
    stride = hparameters['stride']

    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev, hparameters)

    assert (A.shape == (m, n_H, n_W, n_C))

    return A, cache


if __name__ == "__main__":
    np.random.seed(1)
    A_prev = np.random.randn(2, 4, 4, 3)
    hparameters = {"stride": 2, "f": 3}

    A, cache = pool_forward(A_prev, hparameters)
    print("mode = max")
    print("A =", A)
    print()
    A, cache = pool_forward(A_prev, hparameters, mode="average")
    print("mode = average")
    print("A =", A)
