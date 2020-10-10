import numpy as np

def convolve2D(x, kernel, padding=0, strides=1):

    # Gather Shapes of Kernel + Image + Padding
    k_h, k_w, k_in, k_out = kernel.shape
    b_size, x_h, x_w, x_in = x.shape

    # Shape of Output Convolution
    y_h = int(((x_h - k_h + 2 * padding) / strides) + 1)
    y_w = int(((x_w - k_w + 2 * padding) / strides) + 1)
    output = np.zeros((b_size, y_h, y_w, k_out))

    # Apply Equal Padding to All Sides
    for b in range(b_size):
        if padding != 0:
            x_padded = np.zeros((x_h + padding*2, x_w + padding*2, k_in))
            x_padded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = x[b]
            print(x_padded.shape)
        else:
            x_padded = x[b]

        # Iterate through col
        for y in range(output.shape[2]):
            if y % strides == 0:
                for x in range(output.shape[1]):
                    try:
                        # Only Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            for c in range(output.shape[3]):
                                output[b, x, y, c] = (kernel[:,:,:,c] * x_padded[x: x + k_h, y: y + k_w]).sum() 
                    except:
                        break

    return output
