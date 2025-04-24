import functools
import numpy as np
import itasca as it
from itasca import zonearray as za
from itasca import gridpointarray as gpa
import pandas as pd
import os
import time

def run_simulation(mat_prop, freq1, t01):
    # Extract the size of mat_prop
    n = mat_prop.shape[0]

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
    zone create quad size 1 {n} point 0 (0,0) point 1 (5.00,0) point 2 (0,{h_total})
    zone cmodel assign elastic
    """.format(n=n, h_total=n*5))

    # Define material properties
    G = mat_prop[:,2] * mat_prop[:,0]** 2
    K = 2*G*(1+mat_prop[:,1])/3 / (1-2*mat_prop[:,1])

    za.set_prop_scalar("shear", G)
    za.set_prop_scalar("bulk", K)
    za.set_density(mat_prop[:,2])

    # Define boundary conditions
    it.command("""
    zone face skin 
    ; Vicous damping
    zone dynamic damping rayleigh 0.006495192 1.299037873

    ; Boundary Conditions
    zone face apply quiet-tangential range group 'Bottom'
    zone face apply velocity-y 0
    """)

    # Define wavelet
    sigmax = -2 * mat_prop[-1,2] * mat_prop[-1,0]

    # Use str.format() to format the command
    it.command("""
    fish define param
        tt = 15.00
        histdt = 100
        dyndt = 1e-4
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
    it.command("""
    zone history acceleration-x position 2.5 0
    zone history acceleration-x position 2.5 {h_total}
    """.format(h_total=n*5))

    # Run model
    it.command("""
    history interval @histdt
    model dynamic timestep fix @dyndt
    model history name='time' dynamic time-total
    model solve time-total @tt
    """)

    # Save data
    it.command("""
    ; Write Histories
    directory input
    fish define dhwrite
        fname_dhx = 'Data.dat'
        command
            history export [1] [2] vs "time" file @fname_dhx t
        endcommand
    end

    @dhwrite
    """)


# Save results
def save_results(index, path=r"C:\Users\kurt-\Documents\GitHub\GNN_Soil_Dynamics\FLAC\FLAC2D\1D Soil_Bedrock\Results"):
    # Load data
    data = pd.read_csv("Data.dat", sep='\\s+', header=None, skiprows=2)

    # Define the name of file
    name_file = os.path.join(path, 'Data_{:.3f}_.csv'.format(index))

    # Save data
    data.to_csv(name_file, index=False)

if __name__ == '__main__':
    # Define material properties
    Vs_values = np.genfromtxt("model_arrays.csv", delimiter=',', skip_header=1)
    freq1 = 1.5
    t01 = 2.5

    for i in range(len(Vs_values)):
        # Clean the array eliminating nan values
        Vs_array = Vs_values[i][~np.isnan(Vs_values[i])]
        print("Vs_array: ", Vs_array)
        
        # Define the mat_prop array
        mat_prop = np.zeros((Vs_array.size, 3))
        mat_prop[:,0] = Vs_array
        mat_prop[:,1] = 0.3
        mat_prop[:,2] = 2000
        print("Running simulation {}...".format(i))

        start_time = time.time()
        run_simulation(mat_prop, freq1, t01)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Simulation {} completed in {} seconds".format(i, elapsed_time))
        save_results(i)
        break
    
    
    print("All simulations completed!")
        

    # Use %run -i Loop_train.py to run the script