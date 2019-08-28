import tensorflow as tf
import numpy as np
def get_3D_surface_loss(Xdata, Ydata, N_w = 100, N_b = 110):
    W_grid = np.linspace(-2.5,1.5,N_w)
    b_grid = np.linspace(-2.5,1.5,N_b)
    W_grid_mesh, b_grid_mesh = np.meshgrid(W_grid, b_grid)
    
    loss_grid_mesh = 0
    for i in range(Xdata.size):
        x = Xdata[i]
        y = Ydata[i]
        o_grid_i = W_grid_mesh * x + b_grid_mesh
        loss_grid_mesh += np.power(o_grid_i - y,2)
    
    loss_grid_mesh = np.sqrt(loss_grid_mesh/Xdata.size)

    return W_grid_mesh, b_grid_mesh, loss_grid_mesh

def get_training_points(Xdata, Ydata, W_list, b_list, N = 20):
    losses = []
    selected_W = []
    selected_b = []
    
    for i in range (N):
        W = W_list[i]
        b = b_list[i]
        loss_caca = np.sqrt(np.mean(np.power(Xdata* W + b - Ydata,2)))
        losses.append(loss_caca)
        selected_W.append(W)
        selected_b.append(b)

    return selected_W, selected_b, losses
