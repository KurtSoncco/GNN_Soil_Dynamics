import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import importlib.util 
import os
import re

file_path = r"C:\Users\kurt-\Documents\GitHub\GNN_Soil_Dynamics\TF Functions\TTF.py"

# Add the directory containing TTF.py to the system path
sys.path.append(r'C:\Users\kurt-\Documents\GitHub\GNN_Soil_Dynamics\TF Functions')

# Import the module
spec = importlib.util.spec_from_file_location("TTF", file_path)
TTF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TTF)

# Real ground model
def ground_model_data():
    path1 = "./FLAC/central_xacc.dat"
    path2 = "./FLAC/surface_xacc.dat"

    data1 = pd.read_csv(path1, sep='\\s+', header=None, skiprows=2)
    data2 = pd.read_csv(path2, sep='\\s+', header=None, skiprows=2)

    data1.columns = ['0', '1', '2']

    return data1, data2


def figures_making(path_in = "./results/normal/", path_out="./figures/"):

    # Read all the data in the folder path_in
    files = os.listdir(path_in)

    # Separate the files in center and surface
    files_center = [file for file in files if "center" in file]
    files_surface = [file for file in files if "surface" in file]
    files_Vs = [file for file in files if "Vs" in file]

    # Extract the values of index, cv and rh
    pattern = r"[-+]?\d*\.\d+|\d+"
    index_values = np.array([re.findall(pattern, file) for file in files_center]).astype(float)

    # Separate if the index is 0 or 1
    index_0 = np.where(index_values[:,0] == 0)[0] # No extending domain
    index_1 = np.where(index_values[:,0] == 1)[0] # Extending domain

    # Separate the files in center and surface in index 0 and 1
    files_center_0 = [files_center[i] for i in index_0]
    files_center_1 = [files_center[i] for i in index_1]

    files_surface_0 = [files_surface[i] for i in index_0]
    files_surface_1 = [files_surface[i] for i in index_1]

    files_Vs_0 = [files_Vs[i] for i in index_0]
    files_Vs_1 = [files_Vs[i] for i in index_1]

    ground_data1, ground_data2 = ground_model_data()

    # Make the figures
    for i in range(len(files_center_1)):
        print(files_center_1[i])
        fig_center_surface_comparison(files_center_1[i], path_in, path_out, ground_data=ground_data1)
        transfer_function_plots_comparison(files_center_1[i], files_Vs_1[i], path_in, path_out, ground_data1=ground_data1)

        # Surface accelerations
        fig_surface_comparison(files_surface_1[i], path_in, path_out, scale=0.5, ground_data=ground_data2)

        # Vs
        fig_Vs_comparison(files_Vs_0[i], files_Vs_1[i], path_in, path_out)
    

# Define the functions to make the figures
def fig_surface_comparison(files_surface_1, path_in, path_out, scale=0.5, ground_data=None):
    # Read the data using pandas
    data_2 = pd.read_csv(os.path.join(path_in, files_surface_1))

    pattern = r"[-+]?\d*\.\d+|\d+"
    _, cv, rh = re.findall(pattern, files_surface_1) # Since both files have the same values, we can extract from one of them
    cv = float(cv)
    rh = float(rh)

    # Start plotting
    fig, ax = plt.subplots(1,2,figsize=(10,5))

    # First case
    for i, wave in enumerate(data_2.iloc[:, 1:].values.T):
        ax[0].plot(wave*scale + 2.5*i, data_2.iloc[:,0].values, color="gray")

    ax[0].set_xlabel("Position [m]")
    ax[0].set_ylabel("Time [s]")
    ax[0].set_title("Surface acceleration for extensions")
    ax[0].grid(True, which="both")
    
    # Second case
    for i, wave in enumerate(ground_data.iloc[:, 1:].values.T):
        ax[1].plot(wave*scale + 2.5*i, ground_data.iloc[:,0].values, color="gray")

    ax[1].set_xlabel("Position [m]")
    ax[1].set_ylabel("Time [s]")
    ax[1].set_title("Surface acceleration - Base")
    ax[1].grid(True, which="both")

    plt.suptitle("Surface acceleration comparison for CV = {:.3f} and rH = {:.3f}".format(cv, rh))
    plt.savefig(os.path.join(path_out, "surface_comparison_{:.3f}_{:.3f}.png".format(cv, rh)), dpi=300)
    plt.close()

def fig_Vs_comparison(files_Vs_0, files_Vs_1, path_in, path_out, Vs2=1.279259259259259125e+03):
    # Extract data
    pattern = r"[-+]?\d*\.\d+|\d+"
    _, cv, rh = re.findall(pattern, files_Vs_0)  # extract values from one file
    cv = float(cv)
    rh = float(rh)

    # Read the data using pandas
    data0 = pd.read_csv(os.path.join(path_in, files_Vs_0), header=None).values
    data1 = pd.read_csv(os.path.join(path_in, files_Vs_1), header=None).values

    # Transform specific Vs values to nan
    data0[data0 == Vs2] = float('nan')
    data1[data1 == Vs2] = float('nan')

    # Plot the comparison of the Vs with a single colorbar
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im0 = ax[0].imshow(data0, cmap="viridis")
    ax[0].set_title("Vs for no extensions")
    
    im1 = ax[1].imshow(data1, cmap="viridis")
    ax[1].set_title("Vs for extensions")
    
    plt.suptitle("Vs comparison for CV = {:.3f} and rH = {:.3f}".format(cv, rh))
    
    # Add a single colorbar applying to both subplots
    # Here we use the image from the first subplot
    cbar = fig.colorbar(im0, ax=ax.ravel().tolist(), orientation="vertical",
                        fraction=0.05, pad=0.05)
    
    plt.savefig(os.path.join(path_out, "Vs_comparison_{:.3f}_{:.3f}.png".format(cv, rh)), dpi=300)
    plt.close()

def transfer_function_plots_comparison(files_center_1, files_Vs_1, path_in, path_out, ground_data1=None):

    # Assuming the min Vs
    Vs_array = pd.read_csv(os.path.join(path_in, files_Vs_1), header=None).values
    minVs = np.min(Vs_array)

    # Extract data
    pattern = r"[-+]?\d*\.\d+|\d+"
    _, cv, rh = re.findall(pattern, files_center_1) # Since both files have the same values, we can extract from one of them
    cv = float(cv)
    rh = float(rh)

    # Read data
    data_2 = pd.read_csv(os.path.join(path_in, files_center_1))

    # Calculate the TF
    freq_0, TF_0 = TTF.TTF(ground_data1['2'].values, ground_data1['1'].values, Vsmin=minVs)
    freq_1, TF_1 = TTF.TTF(data_2['2'].values, data_2['1'].values, Vsmin=minVs)

    # Plot the comparison of the transfer function
    fig, ax = plt.subplots(figsize=(10,5))
    ax.loglog(freq_0, TF_0, label="Base", linestyle="--")
    ax.loglog(freq_1, TF_1, label="With extensions")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Transfer function")
    ax.legend()
    ax.set_title("Transfer function comparison for CV = {:.3f} and rH = {:.3f}".format(cv, rh))
    ax.grid(True, which="both")
    plt.savefig(os.path.join(path_out, "transfer_function_comparison_{:.3f}_{:.3f}.png".format(cv, rh)), dpi=300)
    plt.close()


def fig_center_surface_comparison(files_center_1, path_in, path_out, ground_data=None):
    # Extract data
    pattern = r"[-+]?\d*\.\d+|\d+"
    _, cv, rh = re.findall(pattern, files_center_1) # Since both files have the same values, we can extract from one of them
    cv = float(cv)
    rh = float(rh)

    # Read data
    data_2 = pd.read_csv(os.path.join(path_in, files_center_1))

    # Plot the comparison of the surface
    fig, ax = plt.subplots(2,1,figsize=(10,5))
    # Surface
    ax[0].plot(ground_data['0'], ground_data['2'], label="Base")
    ax[0].plot(data_2['0'], data_2['2'], label="With extensions")
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Acceleration [m/s^2]")
    ax[0].legend()
    ax[0].set_title("Surface acceleration")
    ax[0].grid(True, which="both")

    # Downhole
    ax[1].plot(ground_data['0'], ground_data['1'], label="Base")
    ax[1].plot(data_2['0'], data_2['1'], label="With extensions")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Acceleration [m/s^2]")
    ax[1].legend()
    ax[1].set_title("Downhole acceleration")
    ax[1].grid(True, which="both")
    plt.suptitle("Comparison of the accelerations for CV = {:.3f} and rH = {:.3f}".format(cv, rh))
    plt.savefig(os.path.join(path_out, "Central_comparison_{:.3f}_{:.3f}.png".format(cv, rh)), dpi=300)
    plt.close()


def comparison_plots(path_in = "./results/normal/", path_out="./figures/"):

    # Read all the data in the folder path_in
    files = os.listdir(path_in)

    # Separate the files in center and surface
    files_center = [file for file in files if "center" in file]

    # Extract the values of index, cv and rh
    pattern = r"[-+]?\d*\.\d+|\d+"
    index_values = np.array([re.findall(pattern, file) for file in files_center]).astype(float)

    # Separate if the index is 0 or 1
    index_1 = np.where(index_values[:,0] == 1)[0] # Extending domain
    index_values_1 = index_values[index_1]

    # Separate the files in center and surface in index 0 and 1
    files_center_1 = [files_center[i] for i in index_1]

    # Base case
    ground_data1, _ = ground_model_data()

    # Compute Ground Transfer function
    freq_0, TF_0 = TTF.TTF(ground_data1['2'].values, ground_data1['1'].values)

    ## Comparison by rH
    rH_values = np.unique(index_values[:,2])
    for j in range(len(rH_values)):
        index_rH = np.where(index_values_1[:,2] == rH_values[j])[0]
        files_center_rH = [files_center_1[i] for i in index_rH]
        cv_vals = np.array([re.findall(pattern, file) for file in files_center_rH]).astype(float)[:,1]
        print(cv_vals)

        # Read the data
        data = [pd.read_csv(os.path.join(path_in, file)) for file in files_center_rH]

        # Compute Transfer function
        TF = [TTF.TTF(data[i]['2'].values, data[i]['1'].values) for i in range(len(data))]

        fig, ax = plt.subplots(figsize=(10,5))
        ax.loglog(freq_0, TF_0, label="Base", linestyle="--")

        for k in range(len(data)):
            ax.loglog(TF[k][0], TF[k][1], label=f"CV = {cv_vals[k]}")

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Transfer function")
        ax.legend()
        ax.set_title("Transfer function comparison for different CV values")
        ax.grid(True, which="both")
        plt.savefig(os.path.join(path_out, f"transfer_function_comparison_rH={rH_values[j]}.png"), dpi=300)
        plt.close()

    ## Comparison by CV
    cv_values = np.unique(index_values[:,1])
    for j in range(len(cv_values)):
        index_cv = np.where(index_values_1[:,1] == cv_values[j])[0]
        files_center_cv = [files_center_1[i] for i in index_cv]
        rH_vals = np.array([re.findall(pattern, file) for file in files_center_cv]).astype(float)[:,2]
        print(rH_vals)

        # Read the data
        data = [pd.read_csv(os.path.join(path_in, file)) for file in files_center_cv]

        # Compute Transfer function
        TF = [TTF.TTF(data[i]['2'].values, data[i]['1'].values) for i in range(len(data))]

        fig, ax = plt.subplots(figsize=(10,5))
        ax.loglog(freq_0, TF_0, label="Base", linestyle="--")

        for k in range(len(data)):
            ax.loglog(TF[k][0], TF[k][1], label=f"rH = {rH_vals[k]}")

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Transfer function")
        ax.legend()
        ax.set_title("Transfer function comparison for different rH values")
        ax.grid(True, which="both")
        plt.savefig(os.path.join(path_out, f"transfer_function_comparison_CV={cv_values[j]}.png"), dpi=300)
        plt.close()



def main():
    figures_making()
    comparison_plots()


if __name__ == "__main__":
    main()
