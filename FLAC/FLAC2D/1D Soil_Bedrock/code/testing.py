import numpy as np
import torch
from scipy.stats import pearsonr
import pandas as pd
import os

def extract_Vs(Vs):
    Vs = Vs[Vs != 0]
    vs_soil = Vs[0]
    h_soil = len(Vs[:-1]) * 5
    vs_bedrock = Vs[-1]
    return vs_soil, vs_bedrock, h_soil

def calculate_pearson_rho(y, y_pred):
    """Calculates Pearson correlation coefficient between y and y_pred.

    Args:
      y: A NumPy array of shape (50, 200, 1) representing the true values.
      y_pred: A NumPy array of shape (50, 200, 1) representing the predicted values.

    Returns:
      A NumPy array of shape (50,) containing the Pearson correlation coefficient for each of the 50 samples.
    """
    rhos = []
    for i in range(y.shape[0]):
        rho, _ = pearsonr(y[i, :, 0], y_pred[i, :, 0])
        rhos.append(rho)
    return np.array(rhos)

def save_and_evaluate_results(model, test_loader, device, Vs_data, save_path):
    model.eval()
    target = []
    predicted = []
    Vs_soil = []
    Vs_bedrock = []
    h_soil = []
    correlation_array = []
    
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            target.append(y.cpu().numpy())
            predicted.append(y_pred.cpu().numpy())
            Vs_soil_i, Vs_bedrock_i, h_soil_i = zip(*[extract_Vs(Vs) for Vs in X[:,0].cpu().numpy()])
            
            # Calculate and print correlation factors
            correlation_i = calculate_pearson_rho(y.cpu().numpy(), y_pred.cpu().numpy())
            
            # Append to lists
            Vs_soil.append(Vs_soil_i)
            Vs_bedrock.append(Vs_bedrock_i)
            h_soil.append(h_soil_i)
            correlation_array.append(correlation_i)

    # Concatenate lists
    target = np.concatenate(target)
    predicted = np.concatenate(predicted)
    Vs_soil = np.concatenate(Vs_soil)
    Vs_bedrock = np.concatenate(Vs_bedrock)
    h_soil = np.concatenate(h_soil)
    corr_array = np.concatenate(correlation_array)
    
    # Save results
    # Create a DataFrame
    data = {
        'Vs_soil': Vs_soil,
        'Vs_bedrock': Vs_bedrock,
        'h_soil': h_soil,
        'correlation': np.squeeze(corr_array)
    }

    # Save DataFrame
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_path, "Correlation_results.csv") , index=False)
    np.save(os.path.join(save_path, "Target.csv"), target)
    np.save(os.path.join(save_path, "Predicted.csv"), predicted)