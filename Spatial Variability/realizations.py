import functools
import numpy as np
#import itasca as it
#from itasca import zonearray as za
#from itasca import gridpointarray as gpa
import pandas as pd
import os
import time
import itertools

def Base_case_extraction(array):
    # Extract unique values
    Vs_unique = np.unique(array)
    Vs1 = Vs_unique[0]
    Vs2 = Vs_unique[1]

    # Extract the height as the number of repetitions of Vs1
    Vs1_count = np.count_nonzero(array == Vs1)
    h = Vs1_count*5
    return Vs1, Vs2, h

def variability(Vs_array, CV, rH, aHV, dx=5, dz=5, extra_z=0):
    # Extract values
    Vs1, Vs2, h = Base_case_extraction(Vs_array)

    # Define the grid
    x = np.linspace(0, 30*dx, 31) + dx/2 # Starting with 31 elements in x direction
    z = np.linspace(0, h+10+extra_z*5, int(h/5+3+extra_z)) + dz/2

    ## Intralayer
    # Create a meshgrid and flatten it
    X, Z = np.meshgrid(x, z)
    X = X.flatten()
    Z = Z.flatten()

    # Vectorized computation of pairwise differences
    tau_x = np.abs(X[:, None] - X[None, :])
    tau_z = np.abs(Z[:, None] - Z[None, :])

    # Compute rho in one go
    rho = np.exp(-2 * (tau_x / rH + tau_z / (rH / aHV)))

    # Compute Cholesky decomposition
    L = np.linalg.cholesky(rho)
    G = L @ np.random.normal(0, 1, len(X))

    # Lognormal transformation
    psi = np.sqrt(np.log(1 + CV ** 2))
    lamb = np.log(Vs1) - 0.5 * psi ** 2

    # Compute the random field
    Z_Vs = np.exp(lamb + psi * G)

    ## Truncated the values
    z_max = np.exp(lamb + psi * 2)
    z_min = np.exp(lamb - psi * 2)
    Z_Vs = np.clip(Z_Vs, z_min, z_max)

    # Reshape the random field
    Z_Vs = Z_Vs.reshape(len(z), len(x))

    ## Interlayer variability
    y_index = np.arange(Z_Vs.shape[0])[:, np.newaxis] * dx

    # Interface model
    freq1 = 0.04
    freq2 = 0.08
    freq3 = 0.102
    offset = 3.440462781459062
    y_add = np.sin(2 * np.pi * freq1 * x + offset) + np.sin(2 * np.pi * freq2 * x+offset) + np.sin(2 * np.pi * freq3 * x+offset)

    # Normalize the signal to have a maximum amplitude of 1
    y_add_normalized = y_add / np.max(np.abs(y_add))
    y_prior = y_add_normalized*dz + h

    # Broadcast and compute the mask: True if row index is less than the y_prior
    mask = y_index < y_prior[np.newaxis, :]

    # Recalculate with the diff the last values
    mask_pos = np.floor(y_prior/dz).astype(int)
    diff = mask_pos*dz + dz/2 - y_prior

    # Mapping function
    mapping = lambda x: dz + x if x < 0 else x
    mapping = np.vectorize(mapping)

    Vs_A = Z_Vs[mask_pos,np.arange(len(x))]
    result_avg = 5/(mapping(diff)/Vs_A + (dz-mapping(diff))/Vs2)

    # Add this to Z_Vs
    Z_Vs[mask_pos,np.arange(len(x))] = result_avg

    # Now mask Z_Vs: assign a fill value (e.g. Vs2) outside the allowed region.
    Z_Vs_masked = np.where(mask, Z_Vs, Vs2)
    X = X.reshape(len(z), len(x))
    Z = Z.reshape(len(z), len(x))

    return Z_Vs_masked, x, z, h

def extending_array(Z_Vs_masked, x, z, dx=5):
    # Let's extract the first and last column
    first_column = Z_Vs_masked[:, 0]
    last_column = Z_Vs_masked[:, -1]
    nx = Z_Vs_masked.shape[1]
    nz = Z_Vs_masked.shape[0]

    # Calculate the number of elements to add
    max_elements_per_side = np.floor((1000 - nz*2 - nx*nz)/nz/2)

    # Copy the columns
    Z_Vs_masked = np.concatenate([np.repeat(first_column[:, np.newaxis], max_elements_per_side, axis=1), Z_Vs_masked, np.repeat(last_column[:, np.newaxis], max_elements_per_side, axis=1)], axis=1)
    x = np.concatenate([np.arange(-(max_elements_per_side)*dx+x[0], 0, dx), x, np.arange(x[-1]+dx, x[-1]+(max_elements_per_side+1)*dx, dx)])
    x = x - np.min(x) + dx/2
    # New grid
    X, Z = np.meshgrid(x, z)

    return Z_Vs_masked, X, Z

def itasca_analysis(Vs_mat, G_mat, K_mat, rho_mat, X, Z, h, freq1=1.5, t01=2.5, dx=5):
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
    zone dynamic damping rayleigh 0.006495192 1.299037873

    ; Boundary Conditions
    zone face apply quiet-tangential range group 'Bottom'
    zone face apply velocity-y 0
    zone dynamic free-field on
    """)

    # Define wavele
    x_mid = np.median(np.unique(X))
    mid_idx = np.argmin(np.abs(X[0,:] - x_mid))
    sigmax = -2 * rho_mat[0, mid_idx] * Vs_mat[0, mid_idx]

    # Use str.format() to format the command
    it.command("""
    fish define param
        tt = 15.00
        histdt = 1
        dyndt = 1.00e-4
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
    """.format(h_total=ny*5, x_mid=x_mid, base_rec=ny*5-(h+10)))

    # Recording of surface accelerations
    x_rec = np.unique(X) - dx/2
    string_rec = "\n".join([
    "zone history acceleration-x position {i} {ny}".format(i=i, ny=Vs_mat.shape[0]*5)
    for i in x_rec
])
    
    it.command(string_rec)

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
        dname_dhx = 'Data2.dat'
        command
            history export [1] [2] vs "time" file @fname_dhx t \n
    """


    extra_commands = "        history export " + " ".join([
    "[{0}]".format(3 + i)
    for i in range(len(x_rec))
]) + " vs \"time\" file @dname_dhx t"
    
    end_command = """
        endcommand
    end

    
    @dhwrite
"""
       
    it.command(base_command + extra_commands + end_command)


    


def save_results(title, path="./results/normal/"):
    # Load data
    data = pd.read_csv("Data.dat", sep='\\s+', header=None, skiprows=2)
    data2 = pd.read_csv("Data2.dat", sep='\\s+', header=None, skiprows=2)

    # Define the name of file
    name_file = os.path.join(path, title+"_center.csv")
    name_file_2 = os.path.join(path, title+"_surface.csv")

    # Save data
    data.to_csv(name_file, index=False)
    data2.to_csv(name_file_2, index=False)


def main_fun():
    # Define the material properties
    Vs = [274.3743743743744,274.3743743743744,274.3743743743744,274.3743743743744,274.3743743743744,274.3743743743744,274.3743743743744,274.3743743743744,274.3743743743744,274.3743743743744,274.3743743743744,274.3743743743744,274.3743743743744,1279.2592592592591]

    # Specify seed
    np.random.seed(1998)
    index = [1]
    CV_list = [0.1, 0.2, 0.3]
    rh_list = [10, 30, 50]

    # Variability
    dx = 5
    dz = 5
    for i, cv, rh in itertools.product(index, CV_list, rh_list):
        print(f"Index: {i}, CV: {cv}, rh: {rh}")
        Z_Vs_masked, x, z, h = variability(Vs, CV=cv, rH=rh, aHV=10, extra_z=0)
    
        # Extending the array
        if i == 1:
            Z_Vs_masked, X, Z = extending_array(Z_Vs_masked, x, z)
        else:
            X, Z = np.meshgrid(x, z)

        X += dx/2
        Z += dz/2
        print(f"Number of elements: ", Z_Vs_masked.shape[0]*Z_Vs_masked.shape[1])
        print(f"Number of elements in x: ", Z_Vs_masked.shape[1])
        print(f"Number of elements in z: ", Z_Vs_masked.shape[0])
        np.savetxt(f"./results/Vs_{i}_{cv}_{rh}.csv", Z_Vs_masked, delimiter=",")

        # Given
        rho = 2000
        nu = 0.3 

        # Develop the material properties
        Vs_mat = Z_Vs_masked[::-1, :]
        G_mat = rho * Vs_mat**2
        K_mat = 2*G_mat*(1+nu)/3 / (1-2*nu)
        rho_mat = np.ones_like(Vs_mat)*rho

        # Start with the Itasca analysis
        itasca_analysis(Vs_mat, G_mat, K_mat, rho_mat, X, Z, h)

        # Save results
        title_format = "Data_{:.3f}_{:.3f}_{:.3f}".format(i, cv, rh)
        save_results(title_format)


    # Use %run -i Loop_train.py to run the script

if __name__ == '__main__':
    main_fun()