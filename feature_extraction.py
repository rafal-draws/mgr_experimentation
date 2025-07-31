import numpy as np
import librosa
import os



def transform_to_ft(frames, metadata_path, normalized: bool):
    ft = []
    
    for i, frame in enumerate(frames):
        ft.append(np.abs(librosa.stft(frame, hop_length=256)))
        
    
    ft = np.array(ft).astype(np.float32)
    
    print(f"Shape of ft: {ft.shape}")
    print(f"Min/max of ft: {np.min(ft)}/{np.max(ft)}")
    
    if normalized:
        ft_mean = np.load(os.path.join(metadata_path, "ft_mean.npy"))
        ft_std = np.load(os.path.join(metadata_path, "ft_std.npy"))
    

        ft = np.log1p(ft)
        
        ft_mean = np.mean(ft, axis=(0, 1), keepdims=True)
        X = (ft - ft_mean) / (ft_std + 1e-8)

    else: 
        X = ft    
    
    print(f"Shape of ft normalized: {X.shape}")
    print(f"Min/max of ft normalized: {np.min(X)}/{np.max(X)}")
            
    return X


def transform_to_spectr(frames, metadata_path, normalized: bool):
    spectr = []
    
    for i, frame in enumerate(frames):
       spectr.append(librosa.amplitude_to_db(
                np.abs(librosa.stft(frame, hop_length=256)), ref=np.max))
    
    spectr = np.array(spectr).astype(np.float32)
    
    
    print(f"Shape of spectrogram: {spectr.shape}")
    print(f"Min/max of spectrogram: {np.min(spectr)}/{np.max(spectr)}")
    
    if normalized:
        spectr_mean = np.load(os.path.join(metadata_path, "spec_mean.npy"))
        spectr_std = np.load(os.path.join(metadata_path, "spec_std.npy"))
            
        spectr = (spectr - spectr_mean) / (spectr_std)
        
        print(f"Shape of spectrogram normalized: {spectr.shape}")
        print(f"Min/max of spectrogram normalized: {np.min(spectr)}/{np.max(spectr)}")
                
    return spectr


def transform_to_mel_spectr(frames, metadata_path, normalized: bool):
    spectr = []
    
    for i, frame in enumerate(frames):
       
        mel_spect = librosa.feature.melspectrogram(
                y=frame, sr=22050, n_fft=8192, hop_length=256, n_mels=1025)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        spectr.append(mel_spect)
    
    spectr = np.array(spectr).astype(np.float32)
    
    
    print(f"Shape of mel spectrogram: {spectr.shape}")
    print(f"Min/max of mel spectrogram: {np.min(spectr)}/{np.max(spectr)}")
    
    if normalized:
        spectr_mean = np.load(os.path.join(metadata_path, "mel_spec_mean.npy"))
        spectr_std = np.load(os.path.join(metadata_path, "mel_spec_std.npy"))
            
        spectr = (spectr - spectr_mean) / (spectr_std)
        
        print(f"Shape of mel spectrogram normalized: {spectr.shape}")
        print(f"Min/max of mel spectrogram normalized: {np.min(spectr)}/{np.max(spectr)}")
                
    return spectr


def transform_to_power_spectr(frames, metadata_path, normalized: bool):

    spectr = []
    
    for i, frame in enumerate(frames):

        ft = librosa.stft(frame, hop_length=256)
        power_spec = np.abs(ft) ** 2
        power_db = librosa.power_to_db(power_spec, ref=np.max)
       
        spectr.append(power_db)
    
    spectr = np.array(spectr).astype(np.float32)
    
    
    print(f"Shape of power spectrogram: {spectr.shape}")
    print(f"Min/max of power spectrogram: {np.min(spectr)}/{np.max(spectr)}")
    
    if normalized:
        spectr_mean = np.load(os.path.join(metadata_path, "power_spec_mean.npy"))
        spectr_std = np.load(os.path.join(metadata_path, "power_spec_std.npy"))
            
        spectr = (spectr - spectr_mean) / (spectr_std)
        
        print(f"Shape of power spectrogram normalized: {spectr.shape}")
        print(f"Min/max of power spectrogram normalized: {np.min(spectr)}/{np.max(spectr)}")
                
    return spectr

def transform_to_mfcc(frames, metadata_path, normalized: bool):

    mfcc = []
    for i, frame in enumerate(frames):

        mfcc_x = librosa.feature.mfcc(
                y=frame, sr=22050, n_mfcc=12, hop_length=256)

        mfcc.append(mfcc_x)
    
    mfcc = np.array(mfcc).astype(np.float32)
    
    
    print(f"Shape of mfcc: {mfcc.shape}")
    print(f"Min/max of mfcc: {np.min(mfcc)}/{np.max(mfcc)}")
    
    if np.min(mfcc) <= -1.0 or np.max(mfcc) >= 1.0:
        print("Data needs normalization")

        if normalized:

            
            mfcc_mean = np.load(os.path.join(metadata_path, "mfcc_mean.npy"))
            mfcc_std = np.load(os.path.join(metadata_path, "mfcc_std.npy"))
            
            mfcc = (mfcc - mfcc_mean) / (mfcc_std)
            
            print(f"Shape of mfcc normalized: {mfcc.shape}")
            print(f"Min/max of mfcc normalized: {np.min(mfcc)}/{np.max(mfcc)}")
                
    return mfcc


def transform_to_chroma(frames, metadata_path, transformation:str, normalized: bool):

    chroma = []
    
    for i, frame in enumerate(frames):

        match transformation:
            case "stft":
                chroma_x = librosa.feature.chroma_stft(
                y=frame, sr=22050, n_chroma=12,
                hop_length=256, n_fft=2048)
            case "cens":
                chroma_x = librosa.feature.chroma_cens(
                y=frame, sr=22050, n_chroma=12,
                hop_length=512)

            case "cqt":
                chroma_x = librosa.feature.chroma_cqt(
                y=frame, sr=22050, n_chroma=12,
                hop_length=512)

        chroma.append(chroma_x)
    
    chroma = np.array(chroma).astype(np.float32)
    
    
    print(f"Shape of chroma {transformation} spectrogram: {chroma.shape}")
    print(f"Min/max of chroma {transformation} spectrogram: {np.min(chroma)}/{np.max(chroma)}")
    
    if np.min(chroma) <= -1.0 or np.max(chroma) >= 1.0:
        print("Data needs normalization")

        if normalized:

            chroma_mean = np.load(os.path.join(metadata_path, f"chroma_{transformation}_mean.npy"))
            chroma_std = np.load(os.path.join(metadata_path, f"chroma_{transformation}_std.npy"))
                
            chroma = (chroma - chroma_mean) / (chroma_std)
            
            print(f"Shape of chroma {transformation} normalized: {chroma.shape}")
            print(f"Min/max of chroma {transformation} normalized: {np.min(chroma)}/{np.max(chroma)}")
                
    return chroma

def transform_to_tonnetz(frames, metadata_path, normalized: bool):

    tonnetz = []
    
    for i, frame in enumerate(frames):

        
        chroma = librosa.feature.chroma_cqt(y=frame, sr=22050)
        tonnetz_x = librosa.feature.tonnetz(chroma=chroma, sr=22050)
        
        tonnetz.append(tonnetz_x)
    
    tonnetz = np.array(tonnetz).astype(np.float32)
    
    
    print(f"Shape of tonnetz: {tonnetz.shape}")
    print(f"Min/max of tonnetz: {np.min(tonnetz)}/{np.max(tonnetz)}")
    
    if np.min(tonnetz) <= -1.0 or np.max(tonnetz) >= 1.0:
        
        print("Data needs normalization")

        if normalized:
            tonnetz_mean = np.load(os.path.join(metadata_path, "tonnetz_mean.npy"))
            tonnetz_std = np.load(os.path.join(metadata_path, "tonnetz_std.npy"))
        
            tonnetz = (tonnetz - tonnetz_mean) / (tonnetz_std)
            
            print(f"Shape of tonnetz normalized: {tonnetz.shape}")
            print(f"Min/max of tonnetz normalized: {np.min(tonnetz)}/{np.max(tonnetz)}")
                
    return tonnetz