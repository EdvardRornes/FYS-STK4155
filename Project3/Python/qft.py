import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the 4x4 gamma matrices in the Dirac representation (using NumPy arrays)
gamma0 = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]], dtype=np.float32)

gamma1 = np.array([[0, 1, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, -1],
                   [0, 0, -1, 0]], dtype=np.float32)

gamma2 = np.array([[0, -1j, 0, 0],
                   [1j, 0, 0, 0],
                   [0, 0, 0, -1j],
                   [0, 0, 1j, 0]], dtype=np.complex64)  # Pauli matrices with complex entries

gamma3 = np.array([[0, 0, 1, 0],
                   [0, 0, 0, -1],
                   [1, 0, 0, 0],
                   [0, -1, 0, 0]], dtype=np.float32)

gamma_matrices = [gamma0, gamma1, gamma2, gamma3]

# Function to generate random spacetime coordinates (t, x, y, z)
def generate_spacetime_coordinates(num_samples):
    t = np.random.uniform(-1, 1, num_samples)
    x = np.random.uniform(-1, 1, num_samples)
    y = np.random.uniform(-1, 1, num_samples)
    z = np.random.uniform(-1, 1, num_samples)
    return np.stack([t, x, y, z], axis=-1)

# Function to generate random spinor field (4-component)
def generate_spinor_field(num_samples):
    return np.random.randn(num_samples, 4).astype(np.float32)

# Update inner_product function to handle complex numbers
def inner_product(psi1, psi2, gamma_matrices):
    # Extract real and imaginary parts separately
    psi1_real = tf.math.real(psi1)
    psi1_imag = tf.math.imag(psi1)
    psi2_real = tf.math.real(psi2)
    psi2_imag = tf.math.imag(psi2)

    # Perform the inner product computation using real and imaginary parts
    real_inner_product = tf.reduce_sum(psi1_real * psi2_real + psi1_imag * psi2_imag)
    imag_inner_product = tf.reduce_sum(psi1_real * psi2_imag - psi1_imag * psi2_real)

    # Combine the real and imaginary parts of the inner product
    inner_prod = tf.complex(real_inner_product, imag_inner_product)
    
    return inner_prod

def compute_loss(psi_pred, spacetime_coords, gamma_matrices):
    ideal_inner_products = []
    
    # Assume spacetime_coords has shape (num_samples, 4)
    # and we expect an ideal inner product of each component w.r.t. spacetime
    for mu in range(4):
        ideal_value = spacetime_coords[:, mu]  # This is a 1D array of size (num_samples,)
        ideal_inner_products.append(ideal_value)

    # Now, psi_pred is a complex tensor of shape (num_samples, 8)
    real_pred = psi_pred[:, :4]  # First 4 components are real parts
    imag_pred = psi_pred[:, 4:]  # Last 4 components are imaginary parts

    # Combine real and imaginary parts
    psi_pred_combined = tf.complex(real_pred, imag_pred)
    
    # Compute the inner product
    current_predictions = []
    for mu in range(4):
        inner_prod = inner_product(psi_pred_combined, psi_pred_combined, gamma_matrices)
        current_predictions.append(inner_prod)
    
    # Now calculate the loss
    loss = 0
    for mu in range(4):
        # Ensure the dimensions match
        loss += tf.reduce_mean(tf.square(current_predictions[mu] - ideal_inner_products[mu]))

    return loss

# Build a simple neural network to predict both real and imaginary parts of the spinor components
def build_nn():
    model = tf.keras.Sequential([ 
        tf.keras.layers.Input(shape=(4,)),  # Input: 4 spacetime coordinates (t, x, y, z)
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        # Output: 4 components of the spinor field (real and imaginary parts for each component)
        tf.keras.layers.Dense(8)  # Output 8 values: real and imaginary parts for each of the 4 components
    ])
    return model

# Train the neural network
def train_nn():
    num_samples = 1000
    spacetime_coords = generate_spacetime_coordinates(num_samples)
    spinor_field = generate_spinor_field(num_samples)

    model = build_nn()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    num_epochs = 1000
    loss_history = []

    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            psi_pred = model(spacetime_coords)
            loss = compute_loss(psi_pred, spacetime_coords, gamma_matrices)
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        loss_history.append(loss.numpy())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.numpy()}")
    
    return model, loss_history

# Function to evaluate the neural network on new test data and plot the results
def evaluate_nn(model):
    num_test_samples = 1000  # Increased number of samples for better surface resolution
    test_spacetime_coords = generate_spacetime_coordinates(num_test_samples)
    true_spinor_field = generate_spinor_field(num_test_samples)

    predictions = model(test_spacetime_coords)
    
    # Extract time, x, y, and z coordinates for plotting
    time_values = test_spacetime_coords[:, 0]  # Time is the first coordinate (t)
    x_values = test_spacetime_coords[:, 1]  # Space is the second coordinate (x)
    y_values = test_spacetime_coords[:, 2]  # Space is the third coordinate (y)
    z_values = test_spacetime_coords[:, 3]  # Space is the fourth coordinate (z)
    
    # Real parts of the predicted spinor components
    real_field_values_0 = np.real(predictions[:, 0])  # Real part of the first component
    real_field_values_1 = np.real(predictions[:, 1])  # Real part of the second component
    real_field_values_2 = np.real(predictions[:, 2])  # Real part of the third component
    real_field_values_3 = np.real(predictions[:, 3])  # Real part of the fourth component

    # Imaginary parts of the predicted spinor components
    imag_field_values_0 = np.imag(predictions[:, 0])  # Imaginary part of the first component
    imag_field_values_1 = np.imag(predictions[:, 1])  # Imaginary part of the second component
    imag_field_values_2 = np.imag(predictions[:, 2])  # Imaginary part of the third component
    imag_field_values_3 = np.imag(predictions[:, 3])  # Imaginary part of the fourth component
    
    # Generate meshgrid for time and space (x, y, z) for plotting
    t_grid, x_grid = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
    
    # Interpolate the field values on the grid for each spatial part
    def interpolate_field_values(field_values, test_coords, spatial_dim):
        field_grid = np.zeros_like(t_grid)
        for i in range(len(t_grid)):
            for j in range(len(x_grid)):
                idx = (np.abs(test_coords[:, 0] - t_grid[i, j]) < 0.1) & \
                      (np.abs(test_coords[:, spatial_dim] - x_grid[i, j]) < 0.1)
                if np.any(idx):
                    field_grid[i, j] = np.mean(field_values[idx])
        return field_grid

    # Interpolate field values for time (t) vs x
    real_field_values_0_grid = interpolate_field_values(real_field_values_0, test_spacetime_coords, 1)
    real_field_values_1_grid = interpolate_field_values(real_field_values_1, test_spacetime_coords, 1)
    real_field_values_2_grid = interpolate_field_values(real_field_values_2, test_spacetime_coords, 2)
    real_field_values_3_grid = interpolate_field_values(real_field_values_3, test_spacetime_coords, 3)
    
    # Interpolate imaginary field values for time (t) vs x
    imag_field_values_0_grid = interpolate_field_values(imag_field_values_0, test_spacetime_coords, 1)
    imag_field_values_1_grid = interpolate_field_values(imag_field_values_1, test_spacetime_coords, 1)
    imag_field_values_2_grid = interpolate_field_values(imag_field_values_2, test_spacetime_coords, 2)
    imag_field_values_3_grid = interpolate_field_values(imag_field_values_3, test_spacetime_coords, 3)

    # Plot the results
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
    fig1.suptitle('Real Parts of Spinor Components', fontsize=16)

    # Real part subplots
    ax1 = axes1[0, 0]
    ax1.plot_surface(t_grid, x_grid, real_field_values_0_grid, cmap='viridis')
    ax1.set_title('Real(ψ₀)')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Space (x)')
    ax1.set_zlabel('Real(ψ₀)')
    
    ax2 = axes1[0, 1]
    ax2.plot_surface(t_grid, x_grid, real_field_values_1_grid, cmap='viridis')
    ax2.set_title('Real(ψ₁)')
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Space (x)')
    ax2.set_zlabel('Real(ψ₁)')
    
    ax3 = axes1[1, 0]
    ax3.plot_surface(t_grid, x_grid, real_field_values_2_grid, cmap='viridis')
    ax3.set_title('Real(ψ₂)')
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Space (x)')
    ax3.set_zlabel('Real(ψ₂)')
    
    ax4 = axes1[1, 1]
    ax4.plot_surface(t_grid, x_grid, real_field_values_3_grid, cmap='viridis')
    ax4.set_title('Real(ψ₃)')
    ax4.set_xlabel('Time (t)')
    ax4.set_ylabel('Space (x)')
    ax4.set_zlabel('Real(ψ₃)')
    
    plt.tight_layout()

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
    fig2.suptitle('Imaginary Parts of Spinor Components', fontsize=16)

    # Imaginary part subplots
    ax5 = axes2[0, 0]
    ax5.plot_surface(t_grid, x_grid, imag_field_values_0_grid, cmap='plasma')
    ax5.set_title('Imag(ψ₀)')
    ax5.set_xlabel('Time (t)')
    ax5.set_ylabel('Space (x)')
    ax5.set_zlabel('Imag(ψ₀)')
    
    ax6 = axes2[0, 1]
    ax6.plot_surface(t_grid, x_grid, imag_field_values_1_grid, cmap='plasma')
    ax6.set_title('Imag(ψ₁)')
    ax6.set_xlabel('Time (t)')
    ax6.set_ylabel('Space (x)')
    ax6.set_zlabel('Imag(ψ₁)')
    
    ax7 = axes2[1, 0]
    ax7.plot_surface(t_grid, x_grid, imag_field_values_2_grid, cmap='plasma')
    ax7.set_title('Imag(ψ₂)')
    ax7.set_xlabel('Time (t)')
    ax7.set_ylabel('Space (x)')
    ax7.set_zlabel('Imag(ψ₂)')
    
    ax8 = axes2[1, 1]
    ax8.plot_surface(t_grid, x_grid, imag_field_values_3_grid, cmap='plasma')
    ax8.set_title('Imag(ψ₃)')
    ax8.set_xlabel('Time (t)')
    ax8.set_ylabel('Space (x)')
    ax8.set_zlabel('Imag(ψ₃)')
    
    plt.tight_layout()
    plt.show()

# Train the neural network
trained_nn = train_nn()

# Evaluate the trained network and plot the results
evaluate_nn(trained_nn)
