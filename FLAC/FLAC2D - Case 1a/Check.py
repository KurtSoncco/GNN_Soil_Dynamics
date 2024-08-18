
# This is a Python script that prints "Hello, World!" to the console.

if __name__ == '__main__':
    
    import functools
    import numpy as np
    import itasca as it
    from itasca import zonearray as za
    from itasca import gridpointarray as gpa
    import pandas as pd

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
    zone create quad size 1 30 point 0 (0,0) point 1 (5.00,0) point 2 (0,150.00)
    zone cmodel assign elastic
    """)

    # Define material properties
    mat_prop = np.genfromtxt(r"C:\Users\kurt-\Documents\GitHub\GNN_Soil_Dynamics\FLAC\FLAC2D - Case 1a\data\data_1.csv", delimiter=",")

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
    # Define the arguments
    freq1 = 1.0
    t01 = 0.5
    sigmax = -2 * mat_prop[-1,2] * mat_prop[-1,0]

    # Use str.format() to format the command
    it.command("""
    fish define wave
    wave = (1-2.0*(math.pi*{freq1}*(dynamic.time.total-{t01}))^2)*math.exp(-((math.pi*{freq1}*(dynamic.time.total-{t01}))^2))
    sigmax = {sigmax}
    histdt = 100
    dyndt = 1.00e-4
    tt = 15.0
    end

    zone face apply stress-shear @sigmax fish @wave range group 'Bottom' 
    """.format(freq1=freq1, t01=t01, sigmax=sigmax))

    # Recording
    it.command("""
    zone history acceleration-x position 2.5 0
    zone history acceleration-x position 2.5 150
    """)

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
        ;
        command
        history export [1] [2] vs "time" file @fname_dhx t
        endcommand
    end

    @dhwrite
    """)

