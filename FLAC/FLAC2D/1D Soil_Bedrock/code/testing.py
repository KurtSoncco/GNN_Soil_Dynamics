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
            correlation_i = pearsonr(y.cpu().numpy(), y_pred.cpu().numpy(), axis=1)[0]
            
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