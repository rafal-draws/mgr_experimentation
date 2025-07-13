import argparse
import logging
import re

import pandas as pd
import utils

# ========== Logging Setup ==========
def setup_logging(log_file="extraction.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


# ========== Helpers ==========
def extract_subgenres(a: str):
    return [int(i) for i in re.findall(r'\d+', a)]


def get_parent_genre(raw_genres, genres_with_subgenres, value_counts):
    subgenres = extract_subgenres(raw_genres)
    for sub in subgenres:
        exists = genres_with_subgenres[genres_with_subgenres['genre_id'].eq(
            sub)]
        if not exists.empty:
            parent = exists['genre_title_parent'].iloc[0]
            if parent in value_counts:
                value_counts[parent] += 1
                return parent
        else:
            return None


def main(metadata_path: str):
    setup_logging()
    logger = logging.getLogger()

    # Load Metadata
    metadata_dir = utils.check_os_get_path(metadata_path)
    logger.info(f"Using metadata path: {metadata_dir}")

    genres = pd.read_csv(f"{metadata_dir}/raw_genres.csv", low_memory=False)
    tracks = pd.read_csv(f"{metadata_dir}/tracks.csv", low_memory=False, skiprows=1)

    # Filter and Join Genre Data
    target_genres = genres[genres['genre_parent_id'].isin([12, 10, 5, 21, 15])]
    genres = genres.rename(columns={"genre_id": "gid"})
    target_genres = target_genres.rename(columns={"genre_parent_id": "gid"})

    genres['gid'] = genres['gid'].astype('Int64') - 1
    target_genres['gid'] = target_genres['gid'].astype('Int64') - 1

    genres_with_subgenres = target_genres.join(
        genres, on="gid", rsuffix="_parent"
    )[
        ['genre_id', 'genre_handle', 'genre_title', 'genre_handle_parent', 'genre_title_parent']
    ]

    # Prepare Initial Valid Tracks
    target_ready_tracks = tracks[tracks['genre_top'].isin(['Rock', 'Electronic', 'Hip-Hop', 'Pop', 'Classical'])]
    logger.info(f"Tracks with labeled main genres: {target_ready_tracks.shape[0]}")

    genre_distribution_pct = (
        target_ready_tracks['genre_top'].value_counts(normalize=True) * 100
    ).round(2).astype(str) + '%'

    logger.info("Class balance in labeled tracks:")
    logger.info(genre_distribution_pct.to_dict())

    logger.info("Track counts per genre:")
    logger.info(target_ready_tracks['genre_top'].value_counts().to_dict())

    # Identify Missing Genre Labels
    unlabeled_tracks = tracks[tracks['genre_top'].isnull() & tracks['genres'].notnull()]
    logger.info(f"Tracks with missing genre_top but available subgenres: {unlabeled_tracks.shape[0]}")

    # Generalise Genres
    genres_with_subgenres.set_index('genre_id')
    value_counts = {g: 0 for g in ['Pop', 'Rock', 'Electronic', 'Hip-Hop', 'Classical']}

    new_genres = unlabeled_tracks.copy()
    new_genres['genre_top'] = unlabeled_tracks['genres_all'].apply(
        lambda raw: get_parent_genre(raw, genres_with_subgenres, value_counts)
    )

    logger.info("New genres derived from subgenre generalisation:")
    logger.info(new_genres['genre_top'].value_counts().to_dict())

    generalised_total = new_genres['genre_top'].value_counts().values.sum()
    logger.info(f"Total tracks reclassified using generalisation: {generalised_total}")

    generalisation_gain_pct = (
        new_genres['genre_top'].value_counts() / unlabeled_tracks.shape[0] * 100
    ).round(2).astype(str) + '%'

    logger.info("Percentage of unlabeled songs recovered by generalisation:")
    logger.info(generalisation_gain_pct.to_dict())

    # Combine Datasets
    df = pd.concat([target_ready_tracks, new_genres], ignore_index=True)
    logger.info("Final genre distribution after combining:")
    logger.info(df['genre_top'].value_counts().to_dict())

    # Generalisation contribution to each class
    contribution_pct = (
        new_genres['genre_top'].value_counts() / df['genre_top'].value_counts() * 100
    ).round(2).astype(str) + '%'

    logger.info("Class gain from generalisation (relative):")
    logger.info(contribution_pct.sort_values(ascending=False).to_dict())

    # Clean Final DataFrame
    cleaned = df[df['genre_top'].notnull()][
        ["Unnamed: 0", "title.1", "name", "bit_rate", "duration", "genre_top"]
    ].copy().rename(columns={"Unnamed: 0": "fma_track_id"})

    utils.save_dataset(artifacts_folder=metadata_dir, a=cleaned, name="tracks_clean")
    logger.info(f"Saved cleaned dataset to {metadata_dir}/tracks_clean.parquet and csv")
    logger.info("Sample of cleaned data:")
    logger.info(cleaned.head().to_dict(orient='records'))

    logger.info("Final class distribution in cleaned dataset:")
    logger.info(cleaned['genre_top'].value_counts().to_dict())


if __name__ == "__main__":
    # dodac mozliwosc wyboru gatunkow
    parser = argparse.ArgumentParser(description="Process FMA metadata and generalise missing genre_top values.")
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to the FMA metadata folder"
    )
    args = parser.parse_args()
    main(args.metadata_path)