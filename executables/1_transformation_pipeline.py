import argparse
import logging
import os

import librosa
import time
import numpy as np
import librosa as li


import pandas as pd
import utils


udf_transformations = []
appends_to_parquet = []
extractions = []

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
def extract_signal_append_parquet(mock, start, end, location, first: bool, signal_length, amount, logger):

    start_time = time.time()

    logger.info(f"Calculating for indexes: {start}, {end}")

    # this extract y udf is the expensive part
    output = mock['path'].iloc[start:end].apply(extract_y_middle)
    output.columns = ["genre_top", "y_minute"]
    output['genre_top'] = mock['genre_top'][start:end]

    output = output.dropna(subset=["y_minute"])
    output = output.rename(columns={"genre_top": "genre", 'y_minute': 'y'})

    elapsed = time.time() - start_time
    logger.info(f"transformation time: {elapsed:.2f}")

    udf_transformations.append(elapsed)

    logger.info("=" * 30)
    logger.info("Saving to Parquet")
    logger.info("=" * 30)

    # append signals.parquet
    if first:
        output.to_parquet(f'{location}/signals_{signal_length}s_{amount}tracks.parquet',
                          engine="fastparquet", compression="snappy")
    else:
        output.to_parquet(f'{location}/signals_{signal_length}s_{amount}tracks.parquet',
                          engine="fastparquet", compression="snappy", append=True)

    elapsed = time.time() - start_time
    appends_to_parquet.append(elapsed)
    logger.info(f"write time: {elapsed:.2f}")


def main(logger, metadata_path: str, output_path: str, data_path: str, signal_lentgth: int = 14, batch_size: int = 128, tracks_amount: int = 50000):

    data_path = utils.check_os_get_path(path=data_path)

    for file in os.listdir(utils.check_os_get_path(metadata_path)):
        if "tracks_clean" in file:
            if file.split(".")[-1] == "parquet":
                tracks_clean = pd.read_parquet(f"{metadata_path}/{file}").astype(
                    {'fma_track_id': 'int64', 'name': 'string', 'genre_top': 'string', 'title.1': 'string'}).set_index("fma_track_id")
            else:
                tracks_clean = pd.read_csv(f"{metadata_path}/{file}").astype({'fma_track_id': 'int64',
                                                                              'name': 'string', 'genre_top': 'string', 'title.1': 'string'}).set_index("fma_track_id")
    logger.info(f"tracks_clean sample: {tracks_clean.head().to_dict()}")

    df = []

    folder_name = os.path.basename(data_path)
    file_list = os.listdir(data_path)

    id_list = [int(x.split(".")[0]) for x in file_list]
    for file_name, file_id in zip(file_list, id_list):
        df.append({
            "folder": folder_name,
            "file": file_name,
            "id": file_id,
            "path": f"{data_path}/{file_name}"
        })

    df = pd.DataFrame(df)
    # # Important, casting the index to same type
    tracks_id_fs_locations = df.rename(columns={"id": "fma_track_id"}).set_index(
        "fma_track_id").astype({'folder': 'string', 'file': 'string', 'path': 'string'})

    tracks_with_paths = tracks_clean.join(
        tracks_id_fs_locations, how="left", rsuffix="tracks_id")
    tracks_with_paths = tracks_with_paths[tracks_with_paths['path'].notna()]
    tracks_with_paths

    genres = list(tracks_with_paths['genre_top'].unique())

    logger.info(f"available genres are: {genres}")
    logger.info(f"available records to process: {tracks_with_paths['genre_top'].value_counts().to_dict()}")
    logger.info(f"SUM: {tracks_with_paths['genre_top'].value_counts().to_dict()}")


    result = pd.concat(tracks_with_paths[tracks_with_paths['genre_top'] == i][:tracks_amount] for i in genres)
    records_chosen_amount = result['genre_top'].value_counts().sum()
    logger.info(f"records that will be processed with provided tracks_amount: {result['genre_top'].value_counts().to_dict()}")
    logger.info(f"in total: {records_chosen_amount}")


    df = result.sort_index()
    df = df[df['duration'].ge(15)]

    # First artifact
    utils.save_dataset(metadata_path, df, "tracks_data_with_paths")

    mock = df.copy(deep=True)

    start_time = time.time()

    for i in range(0, mock.shape[0] // batch_size + 1):
        if i == 0:
            start, end = 0, batch_size
            logger.info(f"Processing batch {i+1}: {start} to {end}")
            logger.info("=" * 30)
            logger.info("Starting batch extraction")
            logger.info("=" * 30)
            extract_signal_append_parquet(
                mock, start, end, output_path, True, signal_lentgth, records_chosen_amount, logger)

        else:
            start, end = i * batch_size, (i+1) * batch_size
            logger.info(f"Processing batch {i+1}: {start} to {end}")
            logger.info("=" * 30)
            logger.info("Starting batch extraction")
            logger.info("=" * 30)
            extract_signal_append_parquet(
                mock, start, end, output_path, False, signal_lentgth, records_chosen_amount, logger)

    end = time.time() - start_time

    logger.info(
        f"data transformation is finished, the data is under {output_path}")

    logger.info(
        f"mp3 to nparray signal extractions avg {np.array(extractions).sum() / len(extractions):.2f}s")

    logger.info("=" * 40)
    logger.info("✅ All done. Summary:")
    logger.info(f"Processed: {mock.shape[0]} tracks")
    logger.info(
        f"Avg. mp3 to nparray signal extractions avg {np.array(extractions).sum() / len(extractions):.2f}s")
    logger.info(
        f"Avg. signal extraction time for a batch of {batch_size}: {np.mean(udf_transformations):.2f}s")
    logger.info(f"Avg. append time: {np.mean(appends_to_parquet):.2f}s")
    logger.info(f"Total time: {int(end // 60)}m{int(end % 60)}s")
    logger.info("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process tracks from .mp3 to float array stored in .parquet")
    parser.add_argument(
        "-m",
        "--metadata-path",
        type=str,
        required=True,
        help="Path to the FMA metadata folder"
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Path the folder which will contain the signals.parquet, containing genre_top and signal columns"
    )
    parser.add_argument(
        "-d",
        "--data-path",
        type=str,
        required=True,
        help="Path to the folder with all the tracks in it (tracks are .mp3 files like 00000.mp3, 000004.mp3, 15624.mp3 etc)"
    )

    parser.add_argument(
        "-s",
        "--signal-length",
        type=int,
        required=False,
        help="length of signal to be extracted -> 14 will provide 14 * 22050 (default frame size), that is 308700 samples. DEFAULT 14"
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        required=True,
        help="amount of tracks for each append cycle"
    )

    parser.add_argument(
        "-t",
        "--tracks-amount",
        type=int,
        required=False,
        help="amount of tracks per genre (DEFAULT = 50000)"
    )

    args = parser.parse_args()

    # HAS TO BE THERE BECAUSE OF SIGNAL PARAMETRIZATION

    setup_logging()
    logger = logging.getLogger()

    def extract_y_middle(path: str):
        """
        takes in path of a file, loads it, and returns a ndarray of samples
        """
        start_time = time.time()

        track = int(path.split("/")[-1].split(".")[0])
        logger.info(f"Progress: {(track / 155278) * 100:.2f}%")
        logger.debug(f"File path: {path}")

        if args.signal_length // 2 > 1:
            hann = 1
        else:
            hann = 0
        try:
            y, sr = librosa.load(path)
            if utils.assert_signal_length(y, sr, args.signal_length):
                try:
                    # extract signal length * 22050
                    y_minute = utils.get_from_middle(y, sr, args.signal_length)

                    # perform hanning window function on 2 seconds from start and the end
                    if hann == 1:
                        y_minute = utils.get_hanned(
                            hann, y_minute, 22050, False)

                    # normalization [0.0:1.0]
                    y_minute = utils.normalize_audio(y_minute)

                    # log purposes
                    elapsed = time.time() - start_time
                    extractions.append(elapsed)
                    logger.info(
                        f"Processed track {track} in {elapsed:.2f} seconds.")

                    return pd.Series([track, y_minute.tolist()])
                except Exception as e:
                    logger.info(f"Error during slicing record {track}: {e}")
                    return pd.Series([track, np.nan])
            else:
                logger.info(f"Record {path} was not long enough.")
                return pd.Series([track, np.nan])
        except Exception as e:
            logger.info(f"Error loading record {path}: {e}")
            return pd.Series([track, np.nan])

    # def extract_f32_from_str(row) -> np.array:
    #     try:
    #         casted      = np.array(row, dtype=np.float32)
    #         print(casted.shape)

    #         return casted
    #     except Exception as e:
    #         print(f"lost in conversion. {e}")
    #         return np.array(np.nan)

    main(logger, args.metadata_path, args.output_path,
         args.data_path, args.signal_length, args.batch_size,
         args.tracks_amount)
