import os
import numpy as np
import librosa
import argparse 
import soundfile as sf

from utils import artifacts_gen



def main(server_data, metadata, song_id, generate_video, generate_features, hop_size, signal_length = 30):

    # op1 
    print("1")
    song    = artifacts_gen.validate_audio_files(server_data, song_id)
    # op2
    
    print("2")
    y       = artifacts_gen.infer_signals(os.path.join(server_data, "uploads", song[0]))
    
    print("3")
    # op3
    y_30    = artifacts_gen.extract_y_middle(y, 30)


    print("4")
    # op4
    # transform frame to .wav for audio gen 
    sample_location = artifacts_gen.generate_audio_from_frames(song_id, y_30, 22050, server_data)
    print(sample_location)
    print("5")
    # op5
    # split to frames for more classifications
    y = artifacts_gen.split_to_frames(y_30, frame_length=22050, hop_length=hop_size)
    
    # op6 todo make async + cache progress to redis
    frames_ft = artifacts_gen.transform_to_ft(y, metadata, True)
    
    if generate_video:
        artifacts_gen.generate_ft_graphs(frames_ft, server_data, song_id)
        artifacts_gen.generate_video(server_data, song_id, "ft", sample_location)
    if generate_features:
        artifacts_gen.save_feature_to_server_data("ft", server_data, song_id, frames_ft) 
    
    # saving the planet
    del frames_ft
    
    # op7
    spectr_normalized = artifacts_gen.transform_to_spectr(y, metadata, True)

    if generate_video:
        artifacts_gen.generate_spectrogram_graphs(spectr_normalized, server_data ,song_id)
        artifacts_gen.generate_video(server_data, song_id, "spectr", sample_location)

    if generate_features:
        artifacts_gen.save_feature_to_server_data("spectr", server_data, song_id, spectr_normalized)
    
    del spectr_normalized

    # op8
    mel_spectr_normalized = artifacts_gen.transform_to_mel_spectr(y, metadata, True)

    if generate_video:
        artifacts_gen.generate_mel_spectrogram_graphs(mel_spectr_normalized, server_data, song_id)
        artifacts_gen.generate_video(server_data, song_id, "mel_spectr", sample_location)


    if generate_features:
        artifacts_gen.save_feature_to_server_data("mel_spectr", server_data, song_id, mel_spectr_normalized)

    del mel_spectr_normalized
    
    # op9
    power_spectr_normalized = artifacts_gen.transform_to_power_spectr(y, metadata, True)

    if generate_features:        
        artifacts_gen.generate_power_spectrogram_graphs(power_spectr_normalized, server_data, song_id)
        artifacts_gen.generate_video(server_data, song_id, "power_spectr", sample_location)

    if generate_features:
        artifacts_gen.save_feature_to_server_data("power_spectr", server_data, song_id, power_spectr_normalized)
    del power_spectr_normalized
    
    # op 10
    mfcc_normalized = artifacts_gen.transform_to_mfcc(y, metadata, True)

    if generate_video:
        artifacts_gen.generate_mfcc_graphs(mfcc_normalized, server_data, song_id)
        artifacts_gen.generate_video(server_data, song_id, "power_spectr", sample_location)

    if generate_features:
        artifacts_gen.save_feature_to_server_data("mfcc", server_data, song_id, mfcc_normalized)

    del mfcc_normalized

    # op 11
    normalized_chroma_stft = artifacts_gen.transform_to_chroma(y, metadata,  "stft", True)

    if generate_video:
        artifacts_gen.generate_chroma_graphs(normalized_chroma_stft, "stft", server_data, song_id)
        artifacts_gen.generate_video(server_data, song_id, "stft", sample_location)

    if generate_features:
        artifacts_gen.save_feature_to_server_data("chroma_stft", server_data, song_id, normalized_chroma_stft)
    
    del normalized_chroma_stft
    
    # op 12
    normalized_chroma_cens = artifacts_gen.transform_to_chroma(y, metadata,  "cens", True)

    if generate_video:
        artifacts_gen.generate_chroma_graphs(normalized_chroma_cens, "cens", server_data, song_id)
        artifacts_gen.generate_video(server_data, song_id, "cens", sample_location)

    if generate_features:
        artifacts_gen.save_feature_to_server_data("chroma_cens", server_data, song_id, normalized_chroma_cens)
    del normalized_chroma_cens
    
    # op 13
    normalized_chroma_cqt = artifacts_gen.transform_to_chroma(y, metadata,  "cqt", True)

    if generate_video:
        artifacts_gen.generate_chroma_graphs(normalized_chroma_cqt, "cqt", server_data, song_id)
        artifacts_gen.generate_video(server_data, song_id, "cqt", sample_location)

    if generate_features:
        artifacts_gen.save_feature_to_server_data("chroma_cqt", server_data, song_id, normalized_chroma_cqt)
    
    # op 14    
    normalized_tonnetz = artifacts_gen.transform_to_tonnetz(y, metadata, True)

    if generate_video:
        artifacts_gen.generate_tonnetz_graphs(normalized_tonnetz, server_data, song_id)
        artifacts_gen.generate_video(server_data, song_id, "tonnetz", sample_location)

    if generate_features:
        artifacts_gen.save_feature_to_server_data("tonnetz", server_data, song_id, normalized_tonnetz)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal Preparation Script")
    parser.add_argument("--server-data", "-d", type=str, required=True, help="server data location")
    parser.add_argument("--metadata", "-m", type=str, required=True, help="Metadata PATH generated from 3_model_training")
    parser.add_argument("--song-id", "-s", type=str, required=True, help="Unique identifier for the song")
    parser.add_argument("--generate-video", "-g", action="store_true", help="Generate video from audio files")
    parser.add_argument("--generate-features", "-f", action="store_true", help="Generate features in .npy format for inference")
    parser.add_argument("--hop-size", required=False, help="Defines hopsize (default 11025). 22050 -> 30frames, 11025 -> 59 frames, 2205 -> 291 frames")
    parser.add_argument("--signal-length", required=False, help="Defines signal length (default 30)")


    args = parser.parse_args()

    main(args.server_data, args.metadata, args.song_id, args.generate_video, args.generate_features, int(args.hop_size), int(args.signal_length))
    
    
    