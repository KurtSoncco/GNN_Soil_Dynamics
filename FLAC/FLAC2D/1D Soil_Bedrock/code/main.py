import torch 
from torch import nn
import numpy as np
from dataloader_TF import get_dataloader, get_data
from training import train_model, test_model
from model import Seq2Seq, EncoderLSTM, DecoderLSTM
from testing import save_and_evaluate_results
import yaml

if __name__ == "__main__":

    # Load yaml file
    config_file_path = "../config/default.yaml"
    config_file = yaml.safe_load(open(config_file_path))

    # File paths
    freq_path = config_file["data"]["freq_path"]
    TF_path = config_file["data"]["target_path"]
    Vs_path = config_file["data"]["data_path"]

    # Get data
    TTF_data, Vs_data, freq_data = get_data(TF_path, Vs_path, freq_path)

    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloader(TTF_data, Vs_data, batch_size=config_file["dataloader"]["batch_size"], 
    training_fraction=config_file["dataloader"]["train_size"])
    X, Y = next(iter(train_loader))

    # Model
    input_size = Vs_data.shape[1]
    hidden_size = config_file["model"]["hidden_size"]
    num_layers = config_file["model"]["num_layers"]
    bidirectional_encoder = config_file["model"]["bidirectional_encoder"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model
    encoder = EncoderLSTM(input_size, hidden_size, num_layers, bidirectional_encoder)
    decoder = DecoderLSTM(hidden_size * 2 if bidirectional_encoder else hidden_size, hidden_size, num_layers, bidirectional_encoder)
    mlp = nn.Linear(hidden_size * 2 if bidirectional_encoder else hidden_size, 1)

    model = Seq2Seq(encoder, decoder, mlp, device)

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_file["optimizer"]["lr"],
                                  weight_decay=config_file["optimizer"]["weight_decay"],
                                  amsgrad=config_file["optimizer"]["amsgrad"], eps=1e-8)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=config_file["scheduler"]["factor"],
                                                              patience=config_file["scheduler"]["patience"], 
                                                              min_lr=config_file["scheduler"]["min_lr"],
                                                              eps=config_file["scheduler"]["eps"])

    print("Prepare for training...")
    # Training
    train_losses, val_losses, model = train_model(model, optimizer, scheduler, train_loader, val_loader, criterion, 
    epochs=config_file["training"]["epochs"]
    , device=device, patience=config_file["training"]["patience"], 
    clipping=config_file["training"]["clipping"], 
    print_epoch=config_file["training"]["print_epoch"])

    # Save model
    torch.save(model.state_dict(), config_file["model"]["save_path"])

    # Save losses
    with open(config_file["testing"]["save_losses"], "w") as f:
        yaml.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    # Test
    test_loss = test_model(model, test_loader, criterion, device=device)

    # Save and evaluate results
    save_and_evaluate_results(model, test_loader, device, Vs_data, config_file["testing"]["save_path"])




