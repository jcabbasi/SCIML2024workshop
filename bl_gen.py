
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import time
import optax
jax.config.update("jax_enable_x64", True)


# Function to calculate the front location at each time step
def BL_x(t, dfw_dsw, qt, A, phi):
    PVI = qt / (A * phi)
    return PVI * dfw_dsw * t

def BL_dfw(t,x,qt,A,phi):
            PVI = qt/(A*phi)
            dfwdsw = x/(t * PVI)
            return dfwdsw

def fw_calc(sw, krwmax, nw, krnwmax, nnw, swc, snwr, muw, munw):
    # Calculate normalized saturation
    S = (sw - swc) / (1 - swc - snwr)
    
    # Calculate relative permeabilities with single saturation exponents
    krw = krwmax * S**nw
    krnw = krnwmax * (1 - S)**nnw
    
    # Calculate mobilities
    lambda_w = krw / muw
    lambda_nw = krnw / munw
    
    # Calculate fractional flow of water
    fw = lambda_w / (lambda_w + lambda_nw)
    
    return fw

def BL_data_gen(qt, A, phi,krwmax, nw, krnwmax, nnw, swc, snwr, muw, munw, x_resol, times):
    # Set up spatial and saturation grid
    # x = torch.linspace(0.001, args.l-0.001, args.x_resol).view(-1, 1)
    sw_values = np.linspace(swc, 1 - snwr, x_resol)

    # Calculate fractional flow and its derivative
    fw = fw_calc(sw_values, krwmax, nw, krnwmax, nnw, swc, snwr, muw, munw)
    # fw_derivative = torch.autograd.grad(fw, sw, torch.ones_like(fw), create_graph=True)[0]
    # fw_second_derivative = torch.autograd.grad(fw_derivative, sw, torch.ones_like(fw), create_graph=True)[0]
    fw_derivative = np.gradient(fw, sw_values)
    fw_second_derivative = np.gradient(fw_derivative, sw_values)
    fw_welge = fw / (sw_values+1e-9)  # Slope from the origin to each point on fw curve

    # Check if there are any valid indices
    # if len(valid_indices) == 0:
    #     raise ValueError("No valid tangent points found. Check parameter values or saturation range.")
    difference_gradient = np.abs(fw_derivative - fw_welge)
    difference_gradient[fw_second_derivative>0]=1000
    # Find the index of the tangent point based on the minimum difference in slopes
    min_slope_index = np.argmin(difference_gradient)
    # min_slope_index = np.where(fw_second_derivative < 0, np.argmin(np.abs(fw_derivative - fw_welge)))[0]

    # Define the tangent point and slope
    sw_tangent = sw_values[min_slope_index]
    fw_tangent = fw_derivative[min_slope_index]
    # fw_derivative[sw_values<sw_tangent]=fw_welge[sw_values<sw_tangent]
    # fw_derivative[sw_values<sw_tangent]=0.0



    # Calculate saturation profile for each time step
    saturation_profiles = []
    x_profiles = []
    dswdxs = []
    for t in times:

        x_front = BL_x(t, fw_derivative, qt, A, phi)
        x_atfront = BL_x(t, fw_tangent, qt, A, phi)
        # dfwdsw_calc = BL_dfw(t, x_values, args.qt, args.A, args.phi)
        
        # f = interpolate.interp1d(fw_derivative, sw_values)
        # Build saturation profile: front saturation before x_front, swc after x_front
        # sw_x = f(dfwdsw_calc)


        saturation_profile = np.where(sw_values > sw_tangent, sw_values, swc)
        x_front = np.where(sw_values >= sw_tangent, x_front, x_atfront+(sw_tangent-sw_values))
        # x_front[x_front>1]=1
        dswdx = np.gradient(saturation_profile, x_front)

        saturation_profiles.append(saturation_profile)
        x_profiles.append(x_front)
        dswdxs.append(dswdx)

    return x_profiles, saturation_profiles, dswdxs, sw_tangent




# Run the data generation function
