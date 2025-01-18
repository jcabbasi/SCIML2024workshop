import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import time
import optax
from bl_gen import BL_data_gen
jax.config.update("jax_enable_x64", True)

""" Simulation using JAX, in differentiable form.
 In this Python file, I tried to follow an object oriented strategy, and follow the same code architecture as in the revised Python file for calculations based on numpy.

 The calculations are still numerical (discritized), but the gradients are calculated and stored. 
 
"""
def xavier_normal_init(key, n_input, n_output):
    stddev = jnp.sqrt(2 / (n_input + n_output))
    return random.normal(key, (n_input, n_output)) * stddev


class parser:

    def __init__(self):
        self.n_x = 50
        self.n_t = 50
        self.lx = 1.0
        self.lt = 0.1 #seconds
        self.dx = self.lx/self.n_x
        self.dt = self.lt/self.n_t
        self.gravity = 0.0

        self.swi = 0.0
        self.pnwi = 10.0 * 1e5 #pascal

        self.muw = 1e-3
        self.munw = 1e-3

        self.krwmax = 1.0
        self.krnwmax = 1.0
        self.nkrw = 2.0
        self.nkrnw = 2.0
        self.snwir = 0.0

        self.swX0 = 1.0
        self.pnwX0 = 10.0 * 1e5 #pascal

        # rock properties
        self.phi = 0.1
        self.k = 1e-12
        self.ut = 1


        


        # Plotting parameters
        self.plot_range = 0.5
        self.plot_every = 2
        self.max_quivers = 21



class BL_JAX:
    def __init__(self,eps=1):
        args = parser()
        self.args = args
        self.x = jnp.arange(args.n_x) * args.dx
        self.t = jnp.arange(args.n_t) * args.dt
        self.sw0  = jnp.ones_like(self.x) * args.swi
        self.pnw0  = jnp.ones_like(self.x) * args.pnwi
        
        self.sw0 = self.sw0.at[0].set(args.swX0)
        self.pnw0 = self.pnw0.at[0].set(args.pnwX0)
        
        self.X, self.T = jnp.meshgrid(self.x, self.t, indexing='ij')
        #first dim: x, second dim: t
        self.sw, self.pnw = jnp.meshgrid(self.sw0, self.pnw0, indexing='ij')
        
        
        self.eps = eps

        # Initialize state variables
        # self.h = self.initialize_height()
        # self.u = jnp.zeros_like(self.h)
        # self.v = jnp.zeros_like(self.h)
        self.cumulative_time = [0.0]
        self.time_now = 0.0


    def apply_boundary_conditions(self, sw, pnw):
        sw = sw.at[0,:].set(self.args.swX0)
        sw = sw.at[:,0].set(self.args.swi)

        pnw = pnw.at[0,:].set(self.args.pnwX0)
        return sw, pnw


    # def initialize_properties(self):
    #     """Initializes the height field with a Gaussian perturbation."""
                
        
        
    #     return self.args.depth + self.eps * jnp.exp(
    #         - (self.X - self.x[self.args.n_x // 2]) ** 2 / self.args.rossby_radius ** 2
    #         - (self.Y - self.y[self.args.n_y - 2]) ** 2 / self.args.rossby_radius ** 2
    #     )
# Using jax.jit compiles the code before execution, significantly reducing run-time by optimizing performance. 
# However, it enforces static evaluation, requiring all array shapes and types to remain consistent.
    
    def fw(self, sw):
        Sw = (sw - self.args.swi) / (1 - self.args.swi - self.args.snwir)

        krw = self.args.krwmax * Sw ** self.args.nkrw
        krnw = self.args.krnwmax * (1-Sw) ** self.args.nkrnw

        fw = krw/self.args.muw / (krw/self.args.muw + krnw/self.args.munw)

        return fw
    
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, sw, pnw):
        sw, pnw = self.apply_boundary_conditions(sw, pnw)

        args = self.args

        #first dim: x, second dim: t
        # dsw_dt = (sw[:,2:]-sw[:,:-2])/args.dt
        # dsw_dt = dsw_dt[1:-1,:]
        
        dsw_dt = (sw[:,1:]-sw[:,:-1])/args.dt
        dsw_dt = dsw_dt[:,:-1]
        dsw_dt = dsw_dt[1:-1,:]



        fw = self.fw(sw)
        # dfw_dx = (fw[2:,:]-fw[:-2,:])/args.dx
        dfw_dx = (fw[1:,:]-fw[:-1,:])/args.dx
        dfw_dx = dfw_dx[:-1,:]
        dfw_dx = dfw_dx[:,1:-1]

        ##
        dsw_dx = (sw[1:,:]-sw[:-1,:])/args.dx
        dsw2_dx2 = (dsw_dx[1:,:]-dsw_dx[:-1,:])/args.dx
        dsw2_dx2 = dsw2_dx2[:,1:-1]
        eps=4e-3


        pde_residual = dsw_dt + args.ut/args.phi * dfw_dx - eps * dsw2_dx2
        pde_residual_unpadded = pde_residual+0.0
        error_mean = jnp.mean(jnp.abs(pde_residual_unpadded)**2)
        


        return error_mean * 100





if __name__ == '__main__':
    # Create argument object and model class
    model = BL_JAX(eps=1.0)
    args = model.args
    tsteps = np.array([10, 20, 30])
    times = tsteps / args.n_t * args.lt

    x_profiles, saturation_profiles, dswdxs, sw_tangent = BL_data_gen(args.ut, 1, args.phi,1, 2, 1, 2, 0, 0, 1e-3, 1e-3, 100, times)

    def loss_fn(sw, pnw):
        error = model.step(sw, pnw)
        return error

    sw = model.sw
    pnw = model.pnw
    # Compute the gradient of the loss function with respect to sw
    grad_loss_fn = jax.grad(loss_fn, argnums=0)
    # Initialize optimizer (e.g., Adam)
    # Define an exponential decay learning rate schedule
    initial_learning_rate = 3e-2
    transition_steps = 5000  # Number of steps over which to decay the learning rate
    decay_rate = 0.98  # Decay factor

    schedule = optax.exponential_decay(
        init_value=initial_learning_rate,
        transition_steps=transition_steps,
        decay_rate=decay_rate
    )

    # Initialize optimizer (e.g., Adam) with the learning rate schedule
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(sw)


        # Initialize the plot
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    line1t, = ax.plot(x_profiles[0], saturation_profiles[0],'k--')
    line2t, = ax.plot(x_profiles[1], saturation_profiles[1],'k--')
    line3t, = ax.plot(x_profiles[2], saturation_profiles[2],'k--')

    line1, = ax.plot(model.x, sw[:, tsteps[0]], label='Initial sw')
    line2, = ax.plot(model.x, sw[:, tsteps[1]], label='Initial sw')
    line3, = ax.plot(model.x, sw[:, tsteps[2]], label='Initial sw')
    ax.set_xlabel('x')
    ax.set_ylabel('sw')
    ax.set_title('Evolution of sw over Time')
    ax.set_xlim([0, args.lx])  # Adjust the range as needed
    ax.grid()
    ax.legend()
    plt.show()



    # # Simulation loop
    num_iterations = 90000
    loss_prev = 0
    for i in range(num_iterations):
        grads = grad_loss_fn(sw, pnw)

        # Update sw using the optimizer
        updates, opt_state = optimizer.update(grads, opt_state)
        sw = optax.apply_updates(sw, updates)
        sw = jnp.clip(sw, model.args.swi, 1-model.args.snwir)  # Clip values to the specified range

        
        # Print loss every 100 iterations
        if i % 1000 == 0:
            loss = loss_fn(sw, pnw)
            current_learning_rate = schedule(i)  # Get the current learning rate
            print(f"Iteration {i}, LR: {current_learning_rate:.5f}, Loss: {loss:.5f}, Change in Loss: {loss - loss_prev:.5f}")
            loss_prev = loss+0.0
            
            # Update the plot
            line1.set_ydata(sw[:, tsteps[0]])  # Update the y-data of the plot
            line2.set_ydata(sw[:, tsteps[1]])  # Update the y-data of the plot
            line3.set_ydata(sw[:, tsteps[2]])  # Update the y-data of the plot
            ax.set_title(f'Evolution of sw over Time (Iteration {i})')
            plt.draw()
            plt.pause(0.01)  # Pause to allow the plot to update


    # Plot the trend of sw over time
    # plt.figure(figsize=(8, 5))
    # #first dim: x, second dim: t
    # timesteps = [0, 10, 20]
    # for t in timesteps:
    #     plt.plot(model.x, sw[:, i],'--', label=f'Timestep {t}')

    # plt.xlabel('Position (x)')
    # plt.ylabel('sw')
    # plt.title('Evolution of sw over Time')
    # plt.legend()
    # plt.grid()
    # plt.show()
    ee=4