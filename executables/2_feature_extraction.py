import argparse
import logging
import os

import librosa
import time
import numpy as np
import librosa as li


import pandas as pd
import utils


# ========== Logging Setup ==========


def setup_logging(log_file="transformation.log"):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


# ========== Helpers ==========
def normalize_and_make_numpy(x, y):
    x = np.array(x).astype(np.float32)
    y = np.array(y).astype(np.int32)

    return x, y


def main(logger, data_location: str, output_path: str):
    files = os.listdir(data_location)
    if not files:
        logger.error(f"No files found in {data_location}")
        return
    for i in files:
        if not i.endswith(".parquet"):
            logger.info(f"File {i} is not a .parquet file")
            continue
        if i.endswith("tracks.parquet"):
            raw = pd.read_parquet(f"{data_location}/{i}")
            break
    else:
        logger.error("No %_tracks.parquet file found in the directory")
        return

    logger.info(f"Loaded tracks data with {len(raw)} entries")
    signals = raw['y']

    # udf register
    one_hot_encoding = np.vectorize(utils.data_transformation.one_hot_function)

    labels = raw['genre']

    # label assigning
    labels = one_hot_encoding(labels)
    logger.info(f"labels shape: {labels.shape}")

    # signals transformation
    signals = signals.apply(lambda x: np.array(
        x[1:-1].split(','), dtype=np.float32))
    signals = signals.to_numpy()

    logger.info(f"signal shape {signals[0].shape}")

    X_train = []
    y_train = []

    frames_amount = signals[1].shape[0] // 22050
    logger.info(f"Frames amount: {frames_amount}")
    # 22050 samples = 2 seconds
    # 11025 samples = 1 second

    [x*11025 for x in range(1, frames_amount * 2 + 1)]
    logger.info(f"X_train shape: {len(X_train)}")
    logger.info(f"y_train shape: {len(y_train)}")

    for signal, label in zip(signals, labels):

        for i in range(0, (frames_amount * 2)-1):

            if i == 0:
                X_train.append(signal[0:22050])
                y_train.append(label)

            else:
                X_train.append(signal[i*11025:(i+2)*11025])
                y_train.append(label)

    for idx, i in enumerate(X_train):
        if i.shape[0] != 22050:
            logger.error(f"idx: {idx} was faulty")

    logger.info(f"Total training signals: {len(X_train)}")
    logger.info(f"Total training labels: {len(y_train)}")

    del labels
    del signals
    del raw

    x, y = normalize_and_make_numpy(X_train, y_train)
    logger.info(f"X_train shape: {x.shape}")
    logger.info(f"y_train shape: {y.shape}")

    x_waveform, y_waveform = x, y
    logger.info(
        f"{x_waveform.shape} {y_waveform.shape}, {x_waveform[0].shape}, {y_waveform[0]}")

    with open(f"{output_path}/labels.npy", "wb") as f:
        np.save(f, y_waveform)
        logger.info("Saved labels to labels.npy")

    with open(f"{output_path}/x_waveform.npy", "wb") as f:
        np.save(f, x_waveform)
        logger.info("Saved labels to x_waveform.npy")

    with open(f"{output_path}/metadata", "a") as f:
        f.write(f"x_waveform.shape={x_waveform.shape}\n")
        f.write(f"labels.shape={y_waveform.shape}\n")

    del x_waveform
    del y_waveform

    ft = []
    ft_y = []

    transformation_time = np.array([])
    op_time = np.array([])

    for idx, (record, label) in enumerate(zip(x, y)):
        start = time.time()
        try:
            transformed = np.abs(librosa.stft(record, hop_length=256))
            fin = time.time() - start

            transformation_time = np.append(transformation_time, fin)

            ft.append(transformed)
            ft_y.append(label)
            op = time.time() - start

            op_time = np.append(op_time, op)


        except Exception as e:
            logger.error(f"Couldn't transform to fourier transform\nException: {e}")


    ft = np.array(ft).astype(np.float32)
    ft_y = np.array(ft_y).astype(np.int32)

    logger.info(f"mean transformation time: {transformation_time.mean():.4f}s")
    logger.info(f"mean operation time: {op_time.mean():.4f}s")

    values, counts = np.unique(ft_y, return_counts=True)
    value_counts = dict(zip(values, counts))
    value_counts, ft.shape, ft_y.shape

    logger.info(f"Fourier Transform shape: {ft.shape}")
    logger.info(f"Fourier Transform labels shape: {ft_y.shape}")
    logger.info(f"Fourier Transform value counts: {value_counts}")


    with open(f"{output_path}/ft.npy", "wb") as f:
        np.save(f, ft)

    with open(f"{output_path}/ft_labels.npy", "wb") as f:
        np.save(f, ft_y)

    with open(f"{output_path}/metadata", "a") as f:
        f.write(f"ft.shape={ft.shape}\n")

    logger.info("Saved Fourier Transforms and labels to ft.npy and ft_labels.npy")
    
    

    del ft
    del ft_y


    # SPECTROGRAMS

    logger.info("Finished extracting Fourier Transforms, now extracting spectrograms...")

    spec_x = []
    spec_y = []

    transformation_time = np.array([])
    op_time = np.array([])

    for idx, (record, label) in enumerate(zip(x, y)):
        start = time.time()
        try:
            spec = librosa.amplitude_to_db(
                np.abs(librosa.stft(record, hop_length=256)), ref=np.max)
            fin = time.time() - start

            transformation_time = np.append(transformation_time, fin)

            spec_x.append(spec)
            spec_y.append(label)
            op = time.time() - start

            op_time = np.append(op_time, op)

            # logger.info(f"{(idx/x.shape[0])*100:.2f}%")

        except:
            logger.info("Couldn't extract spectrogram")

    spec_x = np.array(spec_x).astype(np.float32)
    spec_y = np.array(spec_y).astype(np.int32)

    logger.info(f"mean transformation time: {transformation_time.mean():.4f}s")
    logger.info(f"mean operation time: {op_time.mean():.4f}s")

    with open(f"{output_path}/spectrogram.npy", "wb") as f:
        np.save(f, spec_x)

    with open(f"{output_path}/spectrogram_labels.npy", "wb") as f:
        np.save(f, spec_y)  

    with open(f"{output_path}/metadata", "a") as f:
        f.write(f"spectrogram.shape={spec_x.shape}\n")

    values, counts = np.unique(spec_y, return_counts=True)
    value_counts = dict(zip(values, counts))

    logger.info(f"Spectrogram value counts: {value_counts}")
    logger.info(f"Spectrogram shape: {spec_x.shape}")
    logger.info(f"Spectrogram labels shape: {spec_y.shape}")

    del spec_x
    del spec_y

    logger.info(
        "Finished extracting spectrograms, now extracting mel spectrograms...")
    logger.info("Extracting mel spectrograms...")


    # MEL SPECTROGRAMS

    mel_spec_x = []
    mel_spec_y = []

    transformation_time = np.array([])
    op_time = np.array([])

    for idx, (record, label) in enumerate(zip(x, y)):
        start = time.time()
        try:

            mel_spect = librosa.feature.melspectrogram(
                y=record, sr=22050, n_fft=8192, hop_length=256, n_mels=1025)
            mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

            fin = time.time() - start

            transformation_time = np.append(transformation_time, fin)

            mel_spec_x.append(mel_spect)
            mel_spec_y.append(label)
            op = time.time() - start

            op_time = np.append(op_time, op)

            # logger.info(f"{(idx/x.shape[0])*100:.2f}%")

        except:
            logger.info("Couldn't extract mel spectrogram")

    mel_spec_x = np.array(mel_spec_x).astype(np.float32)
    mel_spec_y = np.array(mel_spec_y).astype(np.int32)

    logger.info(f"mean transformation time: {transformation_time.mean():.4f}s")
    logger.info(f"mean operation time: {op_time.mean():.4f}s")

    values, counts = np.unique(mel_spec_y, return_counts=True)
    value_counts = dict(zip(values, counts))
    value_counts, mel_spec_x.shape, mel_spec_y.shape

    logger.info(f"Mel Spectrogram shape: {mel_spec_x.shape}")
    logger.info(f"Mel Spectrogram labels shape: {mel_spec_y.shape}")
    logger.info(f"Mel Spectrogram value counts: {value_counts}")

    with open(f"{output_path}/mel_spectrogram.npy", "wb") as f:
        np.save(f, mel_spec_x)
    with open(f"{output_path}/mel_spectrogram_labels.npy", "wb") as f:   
        np.save(f, mel_spec_y)

    logger.info("Saved mel spectrograms and labels to mel_spectogram.npy and mel_spectogram_labels.npy")
    with open(f"{output_path}/metadata", "a") as f:
        f.write(f"mel_spectogram.shape={mel_spec_x.shape}\n")

    del mel_spec_x
    del mel_spec_y

    power_spec_x = []
    power_spec_y = []

    transformation_time = np.array([])
    op_time = np.array([])

    for idx, (record, label) in enumerate(zip(x, y)):
        start = time.time()
        try:

            ft = librosa.stft(record, hop_length=256)
            power_spec = np.abs(ft) ** 2

            # zamiana na skalę dB (logarytmiczna skala mocy)
            # S_db = 10 * log10(S/ref)
            # S to moc, ref to wartość odniesienia, np.max(S) to największa moc w całym spektogramie
            power_db = librosa.power_to_db(power_spec, ref=np.max)

            fin = time.time() - start

            transformation_time = np.append(transformation_time, fin)

            power_spec_x.append(power_db)
            power_spec_y.append(label)
            op = time.time() - start

            op_time = np.append(op_time, op)

            # logger.info(f"{(idx/x.shape[0])*100:.2f}%")

        except:
            logger.info("Couldn't extract power spectrogram")

    power_spec_x = np.array(power_spec_x).astype(np.float32)
    power_spec_y = np.array(power_spec_y).astype(np.int32)

    logger.info(f"mean transformation time: {transformation_time.mean():.4f}s")
    logger.info(f"mean operation time: {op_time.mean():.4f}s")

    values, counts = np.unique(power_spec_y, return_counts=True)
    value_counts = dict(zip(values, counts))

    logger.info(f"Power Spectrogram value counts: {value_counts}")
    logger.info(f"Power Spectrogram shape: {power_spec_x.shape}")
    logger.info(f"Power Spectrogram labels shape: {y.shape}")

    with open(f"{output_path}/power_spectrogram.npy", "wb") as f:
        np.save(f, power_spec_x)

    with open(f"{output_path}/power_spectrogram_labels.npy", "wb") as f:
        np.save(f, power_spec_y)
    logger.info(f"Saved power spectrograms and labels to {output_path}/power_spectrogram.npy and {output_path}/power_spectrogram_labels.npy")
    with open(f"{output_path}/metadata", "a") as f:
        f.write(f"power_spectogram.shape={power_spec_x.shape}\n")

    del power_spec_x
    del power_spec_y
    
    #   MFCCs
    
    logger.info(
        "Finished extracting Power Spectrograms, now extracting MFCCs...")



    mfcc = []
    mfcc_y = []

    transformation_time = np.array([])
    op_time = np.array([])

    for idx, (record, label) in enumerate(zip(x, y)):
        start = time.time()
        try:

            mfcc_x = librosa.feature.mfcc(
                y=record, sr=22050, n_mfcc=12, hop_length=256)

            fin = time.time() - start

            transformation_time = np.append(transformation_time, fin)

            mfcc.append(mfcc_x)
            mfcc_y.append(label)

            op = time.time() - start

            op_time = np.append(op_time, op)

            # logger.info(f"{(idx/x.shape[0])*100:.2f}%")

        except:
            logger.info("Couldn't calculate MFCC")

    mfcc = np.array(mfcc).astype(np.float32)
    mfcc_y = np.array(mfcc_y).astype(np.int32)

    logger.info(f"mean transformation time: {transformation_time.mean():.4f}s")
    logger.info(f"mean operation time: {op_time.mean():.4f}s")

    values, counts = np.unique(mfcc_y, return_counts=True)
    value_counts = dict(zip(values, counts))

    logger.info(f"MFCC value counts: {value_counts}")
    logger.info(f"MFCC shape: {mfcc.shape}")
    logger.info(f"MFCC labels shape: {mfcc_y.shape}")

    with open(f"{output_path}/mfcc.npy", "wb") as f:
        np.save(f, mfcc)
    
    with open(f"{output_path}/mfcc_labels.npy", "wb") as f:
        np.save(f, mfcc_y)

    with open(f"{output_path}/metadata", "a") as f:
        f.write(f"mfcc.shape={mfcc.shape}\n")

    del mfcc

    logger.info("Finished extracting MFCCs, now extracting Chroma features...")

    #   CHROMA FEATURES

    chroma = []
    chroma_y = []

    transformation_time = np.array([])
    op_time = np.array([])

    for idx, (record, label) in enumerate(zip(x, y)):
        start = time.time()
        try:

            chroma_x = librosa.feature.chroma_stft(
                y=record, sr=22050, n_chroma=12, hop_length=256, n_fft=2048)

            fin = time.time() - start

            transformation_time = np.append(transformation_time, fin)

            chroma.append(chroma_x)
            chroma_y.append(label)
            op = time.time() - start

            op_time = np.append(op_time, op)

            # logger.info(f"{(idx/x.shape[0])*100:.2f}%")

        except:
            logger.info(f"Couldn't calculate Chroma features for signal {idx}")

    chroma = np.array(chroma).astype(np.float32)
    chroma_y = np.array(chroma_y).astype(np.int32)

    logger.info(f"mean transformation time: {transformation_time.mean():.4f}s")
    logger.info(f"mean operation time: {op_time.mean():.4f}s")

    values, counts = np.unique(chroma_y, return_counts=True)
    value_counts = dict(zip(values, counts))

    logger.info(f"Chroma stft value counts: {value_counts}")
    logger.info(f"Chroma stft shape: {chroma.shape}")
    logger.info(f"Chroma_y stft shape: {y.shape}")

    with open(f"{output_path}/chroma_stft.npy", "wb") as f:
        np.save(f, chroma)

    with open(f"{output_path}/chroma_stft_labels.npy", "wb") as f:
        np.save(f, chroma_y)

    logger.info("Saved chroma stft and labels to chroma_stft.npy and chroma_stft_labels.npy")

    with open(f"{output_path}/metadata", "a") as f:
        f.write(f"chroma_stft.shape={chroma.shape}\n")

    del chroma

    logger.info("Finished extracting Chroma STFT features, now extracting Chroma CQT...")

    #   CHROMA CQT

    chroma = []
    chroma_y = []

    transformation_time = np.array([])
    op_time = np.array([])

    for idx, (record, label) in enumerate(zip(x, y)):
        start = time.time()
        try:

            chroma_cqt_x = librosa.feature.chroma_cqt(
                y=record, sr=22050, n_chroma=12, hop_length=512)

            fin = time.time() - start

            transformation_time = np.append(transformation_time, fin)

            chroma.append(chroma_cqt_x)
            chroma_y.append(label)
            op = time.time() - start

            op_time = np.append(op_time, op)

        except:
            logger.info(f"Couldn't calculate Chroma features for signal {idx}")

    chroma = np.array(chroma).astype(np.float32)

    logger.info(f"mean transformation time: {transformation_time.mean():.4f}s")
    logger.info(f"mean operation time: {op_time.mean():.4f}s")

    values, counts = np.unique(chroma_y, return_counts=True)
    value_counts = dict(zip(values, counts))

    logger.info(f"Chroma cqt value counts: {value_counts}")
    logger.info(f"Chroma cqt shape: {chroma.shape}")
    logger.info(f"Chroma cqt labels shape: {y.shape}")

    with open(f"{output_path}/chroma_cqt.npy", "wb") as f:
        np.save(f, chroma)

    with open(f"{output_path}/chroma_cqt_labels.npy", "wb") as f:
        np.save(f, chroma_y)

    with open(f"{output_path}/metadata", "a") as f:
        f.write(f"chroma_cqt.shape={chroma.shape}\n")


    logger.info("Finished extracting Chroma CQT features, now extracting Chroma CENS...")


    #   CHROMA CENS

    chroma = []
    chroma_y = []

    transformation_time = np.array([])
    op_time = np.array([])

    for idx, (record, label) in enumerate(zip(x, y)):
        start = time.time()
        try:

            chroma_cens_x = librosa.feature.chroma_cens(
                y=record, sr=22050, n_chroma=12, hop_length=512)

            fin = time.time() - start

            transformation_time = np.append(transformation_time, fin)

            chroma.append(chroma_cens_x)
            chroma_y.append(label)
            op = time.time() - start

            op_time = np.append(op_time, op)

            # logger.info(f"{(idx/x.shape[0])*100:.2f}%")

        except:
            logger.info(f"Couldn't calculate Chroma features for signal {idx}")

    chroma = np.array(chroma).astype(np.float32)
    chroma_y = np.array(chroma_y).astype(np.int32)

    logger.info(f"mean transformation time: {transformation_time.mean():.4f}s")
    logger.info(f"mean operation time: {op_time.mean():.4f}s")

    
    values, counts = np.unique(chroma_y, return_counts=True)
    value_counts = dict(zip(values, counts))

    logger.info(f"Chroma Cens value counts: {value_counts}")
    logger.info(f"Chroma Cens shape: {chroma.shape}")
    logger.info(f"Chroma Cens labels shape: {y.shape}")


    with open(f"{output_path}/chroma_cens.npy", "wb") as f:
        np.save(f, chroma)

    with open(f"{output_path}/chroma_cens_labels.npy", "wb") as f:
        np.save(f, chroma_y)
    with open(f"{output_path}/metadata", "a") as f:
        f.write(f"chroma_cens.shape={chroma.shape}\n")


    logger.info("Finished extracting Chroma features, now extracting Tonnetz...")

    #   TONNETZ

    tonnetz_arr = []
    tonnetz_y = []

    transformation_time = np.array([])
    op_time = np.array([])

    for idx, (record, label) in enumerate(zip(x, y)):
        start = time.time()
        try:

            chroma = librosa.feature.chroma_cqt(y=record, sr=22050)
            tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=22050)
            fin = time.time() - start

            transformation_time = np.append(transformation_time, fin)

            tonnetz_arr.append(tonnetz)
            tonnetz_y.append(label)
            op = time.time() - start

            op_time = np.append(op_time, op)

        except:
            logger.error(f"Couldn't calculate Chroma + Tonnetz features for signal {idx}")

    tonnetz_arr = np.array(tonnetz_arr).astype(np.float32)
    tonnetz_y = np.array(tonnetz_y).astype(np.int32)

    logger.info(f"mean transformation time: {transformation_time.mean():.4f}s")
    logger.info(f"mean operation time: {op_time.mean():.4f}s")
    
    values, counts = np.unique(tonnetz_y, return_counts=True)
    value_counts = dict(zip(values, counts))

    logger.info(f"Tonnetz value counts: {value_counts}")
    logger.info(f"Tonnetz shape: {tonnetz_arr.shape}")
    logger.info(f"Tonnetz labels shape: {tonnetz_y.shape}")


    with open(f"{output_path}/tonnetz.npy", "wb") as f:
        np.save(f, tonnetz_arr)

    with open(f"{output_path}/tonnetz_labels.npy", "wb") as f:
        np.save(f, tonnetz_y)
    logger.info("Saved tonnetz and labels to tonnetz.npy and tonnetz_labels.npy")

    with open(f"{output_path}/metadata", "a") as f:
        f.write(f"chroma_cens.shape={tonnetz_arr.shape}\n")


    del tonnetz_arr
    logger.info(f"finished, all data is available in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features from signals.parquet")
    parser.add_argument(
        "-d",
        "--data-location",
        type=str,
        required=True,
        help="Path to the signals.parquet file, which contains the genre_top and signal columns"
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Path the folder which will contain all the features datasets, labels array and metadata file"
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger()

    main(logger, args.data_location, args.output_path)
