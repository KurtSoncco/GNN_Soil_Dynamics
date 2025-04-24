import functools
import numpy as np
import itasca as it
from itasca import zonearray as za
from itasca import gridpointarray as gpa
import pandas as pd
import os
import time
import itertools
#import matplotlib.pyplot as plt

def Base_case_extraction(array):
    # Extract unique values
    Vs_unique = np.unique(array)
    Vs1 = Vs_unique[0]
    Vs2 = Vs_unique[1]

    # Extract the height as the number of repetitions of Vs1
    Vs1_count = np.count_nonzero(array == Vs1)
    h = Vs1_count*5
    h_bedrock = 5 # 5m of bedrock
    return Vs1, Vs2, h, h_bedrock


def itasca_analysis(Vs_mat, G_mat, K_mat, rho_mat, X, Z, h, h_bedrock, freq1=4.16, t01=2.5, dx=1):
    # Extract the size of mat_prop
    ny = Vs_mat.shape[0]
    nx = Vs_mat.shape[1]

    # Start importing commands
    it.command("""
    program load module 'zone'
    program load guimodule 'zone'

    """)

    # Define grid geometry and zones
    it.command("""
    model new
    model configure dynamic
    model large-strain off
    zone create quad size {nx} {ny} point 0 (0,0) point 1 ({xmax},0) point 2 (0,{h_total})
    zone cmodel assign elastic
    """.format(nx=nx, ny=ny, xmax=np.max(X), h_total=np.max(Z)))

    # Define material properties
    za.set_prop_scalar("shear", G_mat.flatten())
    za.set_prop_scalar("bulk", K_mat.flatten())
    za.set_density(rho_mat.flatten())

    # Define boundary conditions
    it.command("""
    zone face skin 
    ; Vicous damping
    ;zone dynamic damping rayleigh 0.006495192 1.299037873
    zone dynamic damping rayleigh 0.006495190529 7.205331360

    ; Boundary Conditions
    zone face apply quiet-tangential range group 'Bottom'
    zone face apply velocity-y 0
    """)

    # Define wavele
    x_mid = np.median(np.unique(X))
    mid_idx = np.argmin(np.abs(X[0,:] - x_mid))
    sigmax = -2 * rho_mat[0, mid_idx] * Vs_mat[0, mid_idx]
    print(f"Vs used in sigmax: {Vs_mat[0, mid_idx]}")

    # Use str.format() to format the command
    it.command("""
    fish define param
        tt = 15.00
        histdt = 1
        dyndt = 1.00e-5
        sigmax = {sigmax}
    end
    @param
    """.format(sigmax=sigmax))

    it.command("""
    fish define wave
        wave = (1-2.0*(math.pi*{freq1}*(dynamic.time.total-{t01}))^2)*math.exp(-((math.pi*{freq1}*(dynamic.time.total-{t01}))^2))
    end
    zone face apply stress-shear @sigmax fish @wave range group 'Bottom'

    """.format(freq1=freq1, t01=t01, sigmax=sigmax))

    # Recording
    x_mid = np.mean(np.concatenate((np.array([np.min(X) - dx]), np.unique(X))))
    it.command("""
    zone history acceleration-x position {x_mid} {base_rec}
    zone history acceleration-x position {x_mid} {h_total}
    """.format(h_total=ny*1, x_mid=x_mid, base_rec=ny*1-(h+h_bedrock)))

    # Run model
    it.command("""
    history interval @histdt
    model dynamic timestep fix @dyndt
    model history name='time' dynamic time-total
    model solve time-total @tt
    """)
    

    # Save data
    base_command = """
    ; Write Histories
    directory input
    fish define dhwrite
        fname_dhx = 'Data.dat'
        command
            history export [1] [2] vs "time" file @fname_dhx t \n
    """
    end_command = """
        endcommand
    end

    @dhwrite
"""
       
    it.command(base_command + end_command)

def save_results(title, path="./results/shallow/"):
    # Load data
    data = pd.read_csv("Data.dat", sep='\\s+', header=None, skiprows=2)

    # Define the name of file
    name_file = os.path.join(path, title+".csv")

    # Save data
    data.to_csv(name_file, index=False)

def base_case(Vs_array, dx, dz, extra_z=5):
    # Extract values
    Vs1, Vs2, h, h_bedrock = Base_case_extraction(Vs_array)
    print(f"Vs1: {Vs1}, Vs2: {Vs2}, h: {h}")

    # Define the grid
    x = np.linspace(0, 1*dx, 1) + dx/2 # 1x1 grid
    z = np.linspace(0, h+h_bedrock+extra_z*dz, int((h+h_bedrock)/dz+1+extra_z)) + dz/2 # 1x1 grid

    # Create a meshgrid and flatten it
    X, Z = np.meshgrid(x, z)
    X = X.flatten()
    Z = Z.flatten()

    # Vs case
    Vs = np.ones_like(Z)*Vs1
    Vs[Z > h] = Vs2
    Z_Vs_masked = Vs.reshape(len(z), len(x))
    X = X.reshape(len(z), len(x))
    Z = Z.reshape(len(z), len(x))
    print(Z_Vs_masked.shape)
    #print(Z_Vs_masked)

    # Given
    rho = 2000
    nu = 0.3 

    # Develop the material properties
    Vs_mat = Z_Vs_masked[::-1, :]
    G_mat = rho * Vs_mat**2
    K_mat = 2*G_mat*(1+nu)/3 / (1-2*nu)
    rho_mat = np.ones_like(Vs_mat)*rho
    X += dx/2
    Z += dz/2

    #freq = 1.5 # Frequency in Hz 
    #freq_list = [freq*0.9, freq, freq*1.1]
    print(f"Vs: {Vs_mat}")

    # First run the analysis
    itasca_analysis(Vs_mat, G_mat, K_mat, rho_mat, X, Z, h, h_bedrock, freq1=4.16, t01=2.5)
    # Save results
    title_format = "Data_modified"
    save_results(title_format, path=".")

    # Start with the Itasca analysis
    """ for i in range(len(freq_list)):
        freq1 = freq_list[i]
        itasca_analysis(Vs_mat, G_mat, K_mat, rho_mat, X, Z, h, h_bedrock, freq1=freq1)

        # Save results
        title_format = "Data_freq" + str(round(freq1,3)) + "_base"
        save_results(title_format, path=".") """


def main_fun():
    # Define the material properties
    Vs = [166.30786881874639,166.30786881874639,821.1231911447858]

    # Specify seed
    np.random.seed(1998)

    # Variability
    dx = 1
    dz = 1

    # Base Case 
    base_case(Vs, dx, dz)

    # Use %run -i Loop_train.py to run the script

if __name__ == '__main__':
    main_fun()