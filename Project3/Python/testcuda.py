import numpy as np
import matplotlib.pyplot as plt 
from numba import cuda

# Mandelbrot calculation
@cuda.jit(device=True)
def mandel(real, imag, iters):
    z_real = real
    z_imag = imag
    for i in range(iters):
        temp_real = z_real * z_real - z_imag * z_imag + real
        z_imag = 2.0 * z_real * z_imag + imag
        z_real = temp_real
        if z_real * z_real + z_imag * z_imag > 4.0:
            return i
    return iters

# Fractal creation kernel
@cuda.jit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    # Determine the pixel position for this thread
    x, y = cuda.grid(2)

    if x < image.shape[1] and y < image.shape[0]:  # Make sure within bounds
        # Map pixel coordinates to complex plane
        pixel_size_x = (max_x - min_x) / image.shape[1]
        pixel_size_y = (max_y - min_y) / image.shape[0]
        real = min_x + x * pixel_size_x
        imag = min_y + y * pixel_size_y
        # Compute Mandelbrot value and set pixel color
        color = mandel(real, imag, iters)
        image[y, x] = color

# Image dimensions
height = 1024
width = 1536

# Allocate the image array on the device
d_image = cuda.device_array((height, width), dtype=np.uint8)

# Define the number of threads per block and blocks per grid
threadsperblock = (16, 16)  # Threads in a block (16x16 threads per block)
blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Launch the kernel
create_fractal[blockspergrid, threadsperblock](-2.0, 1.0, -1.0, 1.0, d_image, 20)

# Copy the result back to the host
image = d_image.copy_to_host()

# Display the fractal image
plt.imshow(image, cmap='hot')
plt.show()
