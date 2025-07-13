import argparse
import logging
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import train_test_split

from utils import *

import numpy as np

def main(logger, data_location, output_path, metadata_path, epochs):

    train_statistics = {}

    def train_and_save(feature, input_shape, model_name):
        train_loader, test_loader = load_data(feature, data_location, metadata_path, logger)
        model, loss_fn, optimizer = define_model(feature, logger)
        
        stat_dict = {}
        trained_model, stat_dict = train_epochs(model, epochs, train_loader, test_loader, loss_fn, optimizer, logger, stat_dict)
        train_statistics[feature] = stat_dict

        trained_model.eval()
        example = torch.rand(*input_shape)
        traced_script_module = torch.jit.trace(trained_model, example)
        traced_script_module.save(f"{output_path}{model_name}.pt")
        logger.info(f"{feature} trained, written to {output_path}{model_name}.pt")

    # logger.info("=== WAVEFORM MODEL ===")
    # train_and_save("waveform", (1, 22050), "wav_model")

    logger.info("=== FOURIER TRANSFORM MODEL ===")
    train_and_save("ft", (1, 1025*87), "ft_model")

    logger.info("=== SPECTROGRAM MODEL ===")
    train_and_save("spectrogram", (1, 1025*87), "spectrogram_model")

    logger.info("=== MEL SPECTROGRAM MODEL ===")
    train_and_save("mel_spectrogram", (1, 1025*87), "mel_spectrogram_model")

    logger.info("=== POWER SPECTROGRAM MODEL ===")
    train_and_save("power_spectrogram", (1, 1025*87), "power_spectrogram_model")

    logger.info("=== MFCC MODEL ===")
    train_and_save("mfcc", (1, 12*87), "mfcc")

    logger.info("=== CHROMA STFT MODEL ===")
    train_and_save("chroma_stft", (1, 12*87), "chroma_stft")

    logger.info("=== CHROMA CQT MODEL ===")
    train_and_save("chroma_cqt", (1, 12*44), "chroma_cqt")

    logger.info("=== CHROMA CENS MODEL ===")
    train_and_save("chroma_cens", (1, 12*44), "chroma_cens")

    logger.info("=== TONNETZ MODEL ===")
    train_and_save("tonnetz", (1, 6*44), "tonnetz")

    logger.info("=== === === === === === === ===")
    logger.info("=== ALL MODELS TRAINED ===")
    logger.info(f"Models and metadata saved to {output_path} and {metadata_path}")

    #output stats
    import json
    stats_path = f"{output_path}training_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(train_statistics, f, indent=4)
    logger.info(f"Training statistics saved to {stats_path}")

def train_epochs(model, epochs, train_ds, test_ds, loss_fn, optimizer, logger, stat_dict=None):
    baseline = 0.0
    target_baseline = 80.0

    test_acc_arr = []
    avg_train_acc_arr = []
    avg_train_loss_arr = []
    acc_gain_arr = []
    baseline_arr = []

    stat_dict = stat_dict if stat_dict is not None else {}
    # For statistics
    stat_dict['stopped_on'] = None
    stat_dict['epoch'] = None
    stat_dict['baseline'] = None
    stat_dict['train_acc'] = None
    stat_dict['test_acc'] = None
    stat_dict['final_train_loss'] = None
    stat_dict['confusion_matrix'] = None
    stat_dict['precision_per_class'] = None
    stat_dict['acc_per_class'] = None
    stat_dict['all_train_acc'] = []
    stat_dict['all_test_acc'] = []
    stat_dict['all_train_loss'] = []
    stat_dict['all_baseline'] = []

    for e in range(epochs):
        logger.info(f"Epoch {e+1}\n{'-'*10}")

        test_acc, avg_train_acc, avg_train_loss, cm, precision_per_class, acc_per_class = train(
            train_ds, model, loss_fn, optimizer, test_ds, logger)

        acc_gain = test_acc - baseline
        baseline = test_acc

        test_acc_arr.append(test_acc)
        avg_train_acc_arr.append(avg_train_acc)
        avg_train_loss_arr.append(avg_train_loss)
        acc_gain_arr.append(acc_gain)
        baseline_arr.append(baseline)

        logger.info(f"Test Accuracy: {test_acc:.2f}%")
        logger.info(f"Train Accuracy: {avg_train_acc:.2f}% - Loss: {avg_train_loss:.4f}")
        logger.info(f"Accuracy gain from baseline: {acc_gain:.2f}%\n")

        # Save stats every epoch
        stat_dict['all_train_acc'].append(avg_train_acc)
        stat_dict['all_test_acc'].append(test_acc)
        stat_dict['all_train_loss'].append(avg_train_loss)
        stat_dict['all_baseline'].append(baseline)

        if baseline >= target_baseline:
            logger.info(f"Test accuracy reached {baseline}! Stopping training.")
            logger.info(f"Last confusion matrix: {cm}")
            logger.info(f"Precision per class:")
            for cls, prec in enumerate(precision_per_class):
                logger.info(f"Class {cls}: {prec * 100:.2f}%")

            stat_dict['stopped_on'] = 'test_baseline'
            stat_dict['epoch'] = e + 1
            stat_dict['baseline'] = baseline
            stat_dict['train_acc'] = avg_train_acc
            stat_dict['test_acc'] = test_acc
            stat_dict['final_train_loss'] = avg_train_loss
            stat_dict['confusion_matrix'] = cm.tolist()
            stat_dict['precision_per_class'] = [float(f"{p:.4f}") for p in precision_per_class]
            stat_dict['acc_per_class'] = [float(f"{p:.2f}") for p in acc_per_class]
            break

        if avg_train_acc >= 99.0:
            logger.info(f"Train accuracy reached 99%! Stopping training to save resources.")
            logger.info(f"Last confusion matrix: {cm}")
            logger.info(f"Precision per class:")
            for cls, prec in enumerate(precision_per_class):
                logger.info(f"Class {cls}: {prec * 100:.2f}%")

            stat_dict['stopped_on'] = 'train_acc'
            stat_dict['epoch'] = e + 1
            stat_dict['baseline'] = baseline
            stat_dict['train_acc'] = avg_train_acc
            stat_dict['test_acc'] = test_acc
            stat_dict['final_train_loss'] = avg_train_loss
            stat_dict['confusion_matrix'] = cm.tolist()
            stat_dict['precision_per_class'] = [float(f"{p:.4f}") for p in precision_per_class]
            stat_dict['acc_per_class'] = [float(f"{p:.2f}") for p in acc_per_class]
            break

    else:
        # If stopped by epochs, fill with last epoch
        stat_dict['stopped_on'] = 'max_epochs'
        stat_dict['epoch'] = epochs
        stat_dict['baseline'] = baseline
        stat_dict['train_acc'] = avg_train_acc_arr[-1] if avg_train_acc_arr else None
        stat_dict['test_acc'] = test_acc_arr[-1] if test_acc_arr else None
        stat_dict['final_train_loss'] = avg_train_loss_arr[-1] if avg_train_loss_arr else None
        stat_dict['confusion_matrix'] = cm.tolist() if 'cm' in locals() else None
        stat_dict['precision_per_class'] = [float(f"{p:.4f}") for p in precision_per_class] if 'precision_per_class' in locals() else None
        stat_dict['acc_per_class'] = [float(f"{p:.2f}") for p in acc_per_class] if 'acc_per_class' in locals() else None

    logger.info("Done!")

    return model, stat_dict




def load_data(feature_name: str, data_path: str, metadata_path: str, logger):

    class PipelineDataset(Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __len__(self):
            return len(self.x)

        def __getitem__(self, index):
            return self.x[index], self.y[index]

    match feature_name:

        case "waveform":
            logger.info("case - waveform")
            with open(f"{data_path}waveform.npy", "rb") as f:
                X = np.load(f)
            with open(f"{data_path}labels.npy", "rb") as f:
                y = np.load(f)

            logger.info(f"Waveform shape: {X.shape}")
            logger.info(f"Waveform labels shape: {y.shape}")
            logger.info(f"X.max, x.min{X.max()}, {X.min()}")

            clean_waveform = X[~np.isnan(X).any(axis=1)]
            clean_waveform_labels = y[~np.isnan(X).any(axis=1)]

            X = clean_waveform
            y = clean_waveform_labels

            del clean_waveform
            del clean_waveform_labels

            logger.info(f"max, min for {feature_name}: {X.max()}, {X.min()}")
            
        case "ft":
            logger.info("case - ft")

            with open(f"{data_path}ft.npy", "rb") as f:
                X = np.load(f)

            with open(f"{data_path}ft_labels.npy", "rb") as f:
                y = np.load(f)


            logger.info(f"Waveform shape: {X.shape}")
            logger.info(f"Waveform labels shape: {y.shape}")

            logger.info(f"X.max, x.min{X.max()}, {X.min()}")

            # loging, normalizing
            logged = np.log1p(X)
            ft_mean = np.mean(logged, axis=(0, 1), keepdims=True)
            
            ft_std = np.std(logged, axis=(0, 1), keepdims=True)
            
            with open(f"{metadata_path}ft_mean.npy", "wb") as f:
                    np.save(f, ft_mean)

            with open(f"{metadata_path}ft_std.npy", "wb") as f:
                    np.save(f, ft_std)

            logger.info(f"ft_mean: {ft_mean}, ft_std: {ft_std}")
            logger.info(f"saved ft_std to {metadata_path}ft_std.npy")
            logger.info(f"saved ft_mean to {metadata_path}ft_mean.npy")
            logger.info(f"ft_mean shape: {ft_mean.shape}, ft_std shape: {ft_std.shape}")
            
            X = (logged - ft_mean) / (ft_std + 1e-8)  # Adding small value to avoid division by zero
            
            logger.info(f"max, min for {feature_name}: {X.max()}, {X.min()}")            
            
        case "spectrogram":
            with open(f"{data_path}spectrogram.npy", "rb") as f:
                X = np.load(f)

            with open(f"{data_path}spectrogram_labels.npy", "rb") as f:
                y = np.load(f)

            logger.info(f"Dataset shape: {X.shape}")
            logger.info(f"Labels shape:{y.shape}")

            logger.info(f"x max = {X.max()}, x min = {X.min()}")


            if X.max() != 1.0 and y.min() != 0.0:
                logger.info("Data needs normalization!!!!")

            spec_mean = np.mean(X, axis=(0, 1), keepdims=True)
            spec_std = np.std(X, axis=(0, 1), keepdims=True)

            with open(f"{metadata_path}spec_mean.npy", "wb") as f:
                    np.save(f, spec_mean)

            with open(f"{metadata_path}spec_std.npy", "wb") as f:
                    np.save(f, spec_std)

            X = (X - spec_mean) / (spec_std)

            logger.info(f"Dataset shape {X.shape}")
            logger.info(f"Labels shape {y.shape}")
            logger.info(f"max/min for {feature_name}: {X.max()}, {X.min()}")
            logger.info(f"{feature_name}_mean shape: {spec_mean.shape}, {feature_name}_std shape: {spec_std.shape}")
            

        case "mel_spectrogram":

            logger.info("case - mel_spectrogram")
            logger.info(f"data_path: {data_path}, metadata_path: {metadata_path}")
            logger.info("Loading mel spectrogram data...")
            with open(f"{data_path}mel_spectrogram.npy", "rb") as f:
                X = np.load(f)

            with open(f"{data_path}mel_spectrogram_labels.npy", "rb") as f:
                y = np.load(f)

            logger.info(f"Dataset shape: {X.shape}")
            logger.info(f"Labels shape:{y.shape}")

            logger.info(f"x max = {X.max()}, x min = {X.min()}")

            
            spec_mean = np.mean(X, axis=(0, 1), keepdims=True)
            spec_std = np.std(X, axis=(0, 1), keepdims=True)

            with open(f"{metadata_path}mel_spec_mean.npy", "wb") as f:
                    np.save(f, spec_mean)

            with open(f"{metadata_path}mel_spec_std.npy", "wb") as f:
                    np.save(f, spec_std)

            X = (X - spec_mean) / (spec_std)

            logger.info(f"Dataset shape {X.shape}")
            logger.info(f"Labels shape {y.shape}")
            logger.info(f"max/min for {feature_name}: {X.max()}, {X.min()}")
            logger.info(f"{feature_name}_mean shape: {spec_mean.shape}, {feature_name}_std shape: {spec_std.shape}")

        case "power_spectrogram":   

            logger.info("case - power_spectrogram")
            logger.info(f"data_path: {data_path}, metadata_path: {metadata_path}")
            logger.info("Loading power spectrogram data...")
            with open(f"{data_path}power_spectrogram.npy", "rb") as f:
                X = np.load(f)

            with open(f"{data_path}power_spectrogram_labels.npy", "rb") as f:
                y = np.load(f)

            logger.info(f"Dataset shape: {X.shape}")
            logger.info(f"Labels shape: {y.shape}")
            
            spec_mean = np.mean(X, axis=(0, 1), keepdims=True)
            spec_std = np.std(X, axis=(0, 1), keepdims=True)

            with open(f"{metadata_path}power_spec_mean.npy", "wb") as f:
                    np.save(f, spec_mean)

            with open(f"{metadata_path}power_spec_std.npy", "wb") as f:
                    np.save(f, spec_std)

            X = (X - spec_mean) / (spec_std)

            logger.info(f"Dataset shape {X.shape}")
            logger.info(f"Labels shape {y.shape}")
            logger.info(f"max/min for {feature_name}: {X.max()}, {X.min()}")
            logger.info(f"{feature_name}_mean shape: {spec_mean.shape}, {feature_name}_std shape: {spec_std.shape}")


        case "mfcc":
            logger.info("case - mfcc")
            with open(f"{data_path}mfcc.npy", "rb") as f:
                X = np.load(f)
            with open(f"{data_path}mfcc_labels.npy", "rb") as f:
                y = np.load(f)
            logger.info(f"Dataset shape: {X.shape}")
            logger.info(f"Labels shape: {y.shape}")
            logger.info(f"x max = {X.max()}, x min = {X.min()}")
            if X.max() != 1.0 and y.min() != 0.0:
                logger.info("Data needs normalization!!!!")
            mfcc_mean = np.mean(X, axis=(0, 1), keepdims=True)
            mfcc_std = np.std(X, axis=(0, 1), keepdims=True)
            with open(f"{metadata_path}mfcc_mean.npy", "wb") as f:
                np.save(f, mfcc_mean)
            with open(f"{metadata_path}mfcc_std.npy", "wb") as f:
                np.save(f, mfcc_std)
            X = (X - mfcc_mean) / (mfcc_std)
            
            logger.info(f"Dataset shape {X.shape}")
            logger.info(f"Labels shape {y.shape}")
            logger.info(f"max/min for {feature_name}: {X.max()}, {X.min()}")
            logger.info(f"{feature_name}_mean shape: {mfcc_mean.shape}, {feature_name}_std shape: {mfcc_std.shape}")
            

        case "chroma_stft":
            logger.info("case - chroma_stft")
            with open(f"{data_path}chroma_stft.npy", "rb") as f:
                X = np.load(f)
            with open(f"{data_path}chroma_stft_labels.npy", "rb") as f:
                y = np.load(f)
            logger.info("Dataset shape:", X.shape)
            logger.info(f"Labels shape: {y.shape}")
            logger.info(f"x max = {X.max()}, x min = {X.min()}")
            if X.max() >= 1.0 or y.min() <= -1.0:
                logger.info("Data needs normalization!!!!")
                chroma_stft_mean = np.mean(X, axis=(0, 1), keepdims=True)
                chroma_stft_std = np.std(X, axis=(0, 1), keepdims=True)
                with open(f"{metadata_path}chroma_stft_mean.npy", "wb") as f:
                    np.save(f, chroma_stft_mean)
                with open(f"{metadata_path}chroma_stft_std.npy", "wb") as f:
                    np.save(f, chroma_stft_std)
                X = (X - chroma_stft_mean) / (chroma_stft_std)
                logger.info(f"{feature_name}_mean shape: {chroma_stft_mean.shape}, {feature_name}_std shape: {chroma_stft_std.shape}")

            logger.info(f"Dataset shape {X.shape}")
            logger.info(f"Labels shape {y.shape}")
            logger.info(f"max/min for {feature_name}: {X.max()}, {X.min()}")
            

        case "chroma_cqt":
            logger.info("case - chroma_cqt")
            with open(f"{data_path}chroma_cqt.npy", "rb") as f:
                X = np.load(f)
            with open(f"{data_path}chroma_cqt_labels.npy", "rb") as f:
                y = np.load(f)
            logger.info(f"Dataset shape: {X.shape}")
            logger.info(f"Labels shape:{y.shape}")
            logger.info(f"x max = {X.max()}, x min = {X.min()}")

            if X.max() >= 1.0 or y.min() <= -1.0:
                logger.info("Data needs normalization!!!!")
            
                chroma_cqt_mean = np.mean(X, axis=(0, 1), keepdims=True)
                chroma_cqt_std = np.std(X, axis=(0, 1), keepdims=True)
                
                with open(f"{metadata_path}chroma_cqt_mean.npy", "wb") as f:
                    np.save(f, chroma_cqt_mean)
                with open(f"{metadata_path}chroma_cqt_std.npy", "wb") as f:
                    np.save(f, chroma_cqt_std)
                
                X = (X - chroma_cqt_mean) / (chroma_cqt_std)
                logger.info("Chroma CQT mean/std:", chroma_cqt_mean, chroma_cqt_std)
            
            logger.info("Dataset shape:", X.shape)
            logger.info(f"Labels shape: {y.shape}")
            logger.info("Chroma CQT max/min:", X.max(), X.min())
            
        case "chroma_cens":
            logger.info("case - chroma_cens")
            with open(f"{data_path}chroma_cens.npy", "rb") as f:
                X = np.load(f)
            with open(f"{data_path}chroma_cens_labels.npy", "rb") as f:
                y = np.load(f)
            logger.info("Dataset shape:", X.shape)
            logger.info(f"Labels shape: {y.shape}")
            logger.info(f"x max = {X.max()}, x min = {X.min()}")
            if X.max() >= 1.0 or y.min() <= -1.0:
                logger.info("Data needs normalization!!!!")
                chroma_cens_mean = np.mean(X, axis=(0, 1), keepdims=True)
                chroma_cens_std = np.std(X, axis=(0, 1), keepdims=True)
                with open(f"{metadata_path}chroma_cens_mean.npy", "wb") as f:
                    np.save(f, chroma_cens_mean)
                with open(f"{metadata_path}chroma_cens_std.npy", "wb") as f:   
                    np.save(f, chroma_cens_std)
                X = (X - chroma_cens_mean) / (chroma_cens_std)
                logger.info(f"{feature_name}_mean shape: {chroma_cens_mean.shape}, {feature_name}_std shape: {chroma_cens_std.shape}")
    
            logger.info(f"Dataset shape {X.shape}")
            logger.info(f"Labels shape {y.shape}")
            logger.info(f"max/min for {feature_name}: {X.max()}, {X.min()}")
            
        case "tonnetz":
            logger.info("case - tonnetz")
            with open(f"{data_path}tonnetz.npy", "rb") as f:
                X = np.load(f)
            with open(f"{data_path}tonnetz_labels.npy", "rb") as f:
                y = np.load(f)
            logger.info("Dataset shape:", X.shape)
            logger.info(f"Labels shape: {y.shape}")
            logger.info(f"x max = {X.max()}, x min = {X.min()}")
            if X.max() >= 1.0 or y.min() <= -1.0:
                logger.info("Data needs normalization!!!!")
                tonnetz_mean = np.mean(X, axis=(0, 1), keepdims=True)
                tonnetz_std = np.std(X, axis=(0, 1), keepdims=True)
                with open(f"{metadata_path}tonnetz_mean.npy", "wb") as f:
                    np.save(f, tonnetz_mean)
                with open(f"{metadata_path}tonnetz_std.npy", "wb") as f:
                    np.save(f, tonnetz_std)
                X = (X - tonnetz_mean) / (tonnetz_std)
                logger.info("Tonnetz mean/std:", tonnetz_mean, tonnetz_std)
        
            logger.info("Dataset shape:", X.shape)
            logger.info(f"Labels shape: {y.shape}")
            logger.info("Tonnetz max/min:", X.max(), X.min())
            

        case _:
            logger.error("bad code, wrong case")
            return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.95, random_state=42
    )

    logger.info(
        f"{X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")

    train_ds = PipelineDataset(X_train, y_train)
    test_ds = PipelineDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=True)

    return train_loader, test_loader




def define_model(model_name: str, logger):
    match model_name:
        case "waveform":
            first_layer = nn.Linear(22050, 512)
            logger.info(f"{model_name} chosen for first layer, {first_layer}")
        case "ft":
            first_layer = nn.Linear(1025 * 87, 512)
            logger.info(f"{model_name} chosen for first layer, {first_layer}")
        case "spectrogram":
            first_layer = nn.Linear(1025 * 87, 512)
            logger.info(f"{model_name} chosen for first layer, {first_layer}")
        case "mel_spectrogram":
            first_layer = nn.Linear(1025 * 87, 512)
            logger.info(f"{model_name} chosen for first layer, {first_layer}")
        case "power_spectrogram":
            first_layer = nn.Linear(1025 * 87, 512)
            logger.info(f"{model_name} chosen for first layer, {first_layer}")
        case "mfcc":
            first_layer = nn.Linear(12 * 87, 512)
            logger.info(f"{model_name} chosen for first layer, {first_layer}")
        case "chroma_stft":
            first_layer = nn.Linear(12 * 87, 512)
            logger.info(f"{model_name} chosen for first layer, {first_layer}")
        case "chroma_cqt":
            first_layer = nn.Linear(12 * 44, 512)
            logger.info(f"{model_name} chosen for first layer, {first_layer}")
        case "chroma_cens":
            first_layer = nn.Linear(12 * 44, 512)
            logger.info(f"{model_name} chosen for first layer, {first_layer}")
        case "tonnetz":
            first_layer = nn.Linear(6 * 44, 512)
            logger.info(f"{model_name} chosen for first layer, {first_layer}")

    class PipelineModel(nn.Module):
        def __init__(self):
            super(PipelineModel, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                first_layer,
                nn.ReLU(),
                nn.Linear(512,5),
            )

        def forward(self, x):
            if model_name != "waveform":
                x = x.view(x.size(0), -1) 
            logits = self.linear_relu_stack(x)
            return logits
    
    model = PipelineModel()

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr = 0.0001, weight_decay=1e-5)

    logger.info(model)
    logger.info(loss_function)
    logger.info(optimizer)

    return model, loss_function, optimizer
    



def save_model(model, output_path, model_name):
    
    path = f"{output_path}{model_name}.pt"
    torch.save(model.state_dict(), path)

    return path


def get_accuracy(pred, labels):
    _, predictions = torch.max(pred, 1)
    correct = (predictions == labels).float().sum()
    accuracy = correct / labels.shape[0]
    return accuracy.item() * 100


def test(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_function, logger):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            y = y.long()
            loss = loss_function(pred, y)
            test_loss += loss.item()

            preds = pred.argmax(1)
            correct += (preds == y).type(torch.int).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss /= num_batches
    accuracy = correct / size * 100

    # logger.info(
    #     f"Test error:\n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f}")

    cm = confusion_matrix(all_labels, all_preds)
    # logger.info("\nConfusion Matrix:")
    # logger.info(cm)

    # logger.info("\nPer-class accuracy:")
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_per_class = 100 * cm.diagonal() / cm.sum(axis=1)
        acc_per_class = np.nan_to_num(acc_per_class)  # replace NaN with 0

    # for cls, acc in enumerate(acc_per_class):
        # logger.info(f"Class {cls}: {acc:.2f}%")

    precision_per_class = precision_score(
        all_labels, all_preds, average=None, zero_division=0)
    # logger.info("\nPer-class precision:")
    # for cls, prec in enumerate(precision_per_class):
    #     logger.info(f"Class {cls}: {prec * 100:.2f}%")

    return accuracy, cm, precision_per_class, acc_per_class


def train(dataloader: torch.utils.data.DataLoader,
          model: torch.nn.Module,
          loss_fn,
          optimizer,
          test_loader,
          logger):
    model.train()
    total_acc = 0
    total_loss = 0
    num_batches = len(dataloader)

    for batch, (X, y) in enumerate(dataloader):
        y = y.long()
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Tracking
        total_acc += get_accuracy(pred, y)
        total_loss += loss.item()

    avg_train_acc = total_acc / num_batches
    avg_train_loss = total_loss / num_batches

    test_acc, cm, precision_per_class, acc_per_class = test(test_loader, model, loss_fn, logger)


    return test_acc, avg_train_acc, avg_train_loss, cm, precision_per_class, acc_per_class




def setup_logging(log_file="transformation.log"):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features from signals.parquet")
    parser.add_argument(
        "-d",
        "--data-path",
        type=str,
        required=True,
        help="Path to the .npy data (waveform.npy, waveform_labels.npy)"
    )
    parser.add_argument(
        "-m",
        "--metadata-path",
        type=str,
        required=True,
        help="training metadata path (will produce log file)"
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Path the folder which will contain all the ML models and transformation artifacts (means/stds for normalization)."
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        required=True,
        help="Count of epochs of training."
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger()

    main(logger, args.data_path, args.output_path, args.metadata_path, args.epochs)