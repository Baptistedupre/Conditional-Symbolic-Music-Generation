import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import note_seq
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Tuple
import pandas as pd
from joblib import Parallel, delayed
import random

from utils.melody_converter import melody_2bar_converter
from utils import songs_utils
from models.music_vae.model import MusicVAE
import warnings

# Suppress the specific FutureWarning from torch.load
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")


def collate_fn(batch):
    """Collate function to properly batch tensors from MIDIDataset."""
    # Stack all tensors and features from the batch, converting to float32
    tensors = torch.stack([item["tensor"].to(torch.float32) for item in batch])
    features = torch.stack([item["feature"].to(torch.float32) for item in batch])
    paths = [item["path"] for item in batch]

    return {"tensor": tensors, "feature": features, "path": paths}


class MIDIDataset(Dataset):
    """Dataset for loading preprocessed MIDI files."""

    def __init__(
        self,
        data_dir: str = "data/songs/",
        feature_type: Literal["OneHotGenre", "Features"] = "OneHotGenre",
    ):
        """
        Args:
            data_dir: Directory containing preprocessed .npy files
            feature_type: Which feature to use ('OneHotGenre' or 'Features')
        """
        self.data_dir = data_dir
        self.feature_type = feature_type
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".pt")]

        if not self.file_list:
            raise RuntimeError(f"No .pt files found in {data_dir}")

    def __len__(self):
        return (
            len(self.file_list) * 100
        )  # Arbitrary multiplier to allow multiple samples per file

    def __getitem__(self, _):  # idx is ignored as we're sampling randomly
        max_attempts = 100  # Prevent infinite loops
        for attempt in range(max_attempts):
            # Choose a random file
            filename = random.choice(self.file_list)
            file_path = os.path.join(self.data_dir, filename)

            # Load the data using torch.load with error handling
            # Check if file exists before trying to load
            if not os.path.exists(file_path):
                self.file_list.remove(filename)
                continue

            # Load the data using torch.load with error handling
            try:
                data_dict = torch.load(file_path, weights_only=True)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                if os.path.exists(file_path):
                    os.remove(file_path)  # Remove file if it's corrupted
                self.file_list.remove(filename)
                continue

            # Get feature tensor based on specified type
            feature = data_dict[self.feature_type]

            # Choose a random melody
            melody_keys = [k for k in data_dict.keys() if k.startswith("melody_")]
            if not melody_keys:  # Skip if no melodies
                if os.path.exists(file_path):
                    os.remove(file_path)  # Remove file if no melodies
                self.file_list.remove(filename)
                print(f"Removed empty file. {filename}")
                continue

            melody_key = random.choice(melody_keys)
            tensors = data_dict[melody_key]

            # Check if the melody has any tensors
            if len(tensors) == 0:  # Skip if empty melody
                # Remove melody from data_dict
                data_dict.pop(melody_key)
                # Save the updated data_dict
                torch.save(data_dict, file_path, _use_new_zipfile_serialization=True)
                continue

            tensor = random.choice(tensors)
            return {"tensor": tensor, "feature": feature, "path": filename}

        raise RuntimeError("Could not find valid melody after multiple attempts")

    @staticmethod
    def get_dataloader(dataset, batch_size=64, shuffle=True, num_workers=4):
        """Creates a DataLoader for the dataset."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )


def convert_midi_to_tensors(
    midi_path: str, chunk_size: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a MIDI file to input, output and control tensors using Song class."""
    try:
        # Load and parse MIDI file
        ns = note_seq.midi_file_to_note_sequence(midi_path)

        # Extract melodies using song_utils
        melodies = songs_utils.extract_melodies_raw(ns)
        if not melodies:
            return None

        # Create Song objects for each melody
        songs = [
            songs_utils.Song(melody, melody_2bar_converter, chunk_size)
            for melody in melodies
        ]

        # Get tensors using Song.chunks method
        all_tensors = {}
        for i, song in enumerate(songs):
            tensors, _ = song.chunks()
            # Convert to float8 format for memory efficiency
            torch_tensors = torch.tensor(tensors).to(torch.float8_e4m3fn)
            all_tensors[f"melody_{i}"] = torch_tensors

        return all_tensors

    except Exception as e:
        print(f"Error processing {midi_path}: {str(e)}")
        return None


def preprocess_and_save_midi(
    midi_path: str, row: pd.Series, save_dir: str = "data/songs/"
):
    """Process MIDI file and save tensors to disk using compressed PyTorch format."""
    try:
        tensors = convert_midi_to_tensors(midi_path)
        if tensors is None:
            return False

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Prepare data dictionary with both feature types
        data_dict = tensors
        # Convert features to float8 as well
        data_dict["OneHotGenre"] = torch.tensor(row["OneHotGenre"]).to(
            torch.float8_e4m3fn
        )
        data_dict["Features"] = torch.tensor(row["Features"]).to(torch.float8_e4m3fn)

        # Clean filename by removing .mid extension before adding .pt
        midi_name = os.path.basename(midi_path)
        filename = midi_name.replace(".mid", "") + ".pt"
        save_path = os.path.join(save_dir, filename)
        torch.save(data_dict, save_path, _use_new_zipfile_serialization=True)
        return True

    except Exception as e:
        print(f"Error processing {midi_path}: {str(e)}")
        return False


def create_preprocessed_dataset(matching_path: str):
    """Preprocess all MIDI files and save them to disk."""
    matching_df = pd.read_json(matching_path)

    # Process MIDI files in parallel
    Parallel(n_jobs=-1)(
        delayed(preprocess_and_save_midi)(row["Path"], row)
        for _, row in matching_df.iterrows()
    )


class EmbeddingDataset(Dataset):
    """Dataset for loading preprocessed MIDI files."""

    def __init__(
        self,
        data_dir: str = "data/songs_embeddings/",
        feature_type: Literal["OneHotGenre"] = "OneHotGenre",
    ):
        """
        Args:
            data_dir: Directory containing preprocessed .npy files
            feature_type: Which feature to use ('OneHotGenre' or 'Features')
        """
        self.data_dir = data_dir
        self.feature_type = feature_type
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".pt")]

        if not self.file_list:
            raise RuntimeError(f"No .pt files found in {data_dir}")

    def __len__(self):
        return (
            len(self.file_list) * 100
        )  # Arbitrary multiplier to allow multiple samples per file

    def __getitem__(self, _):  # idx is ignored as we're sampling randomly
        max_attempts = 100  # Prevent infinite loops
        for attempt in range(max_attempts):
            # Choose a random file
            filename = random.choice(self.file_list)
            file_path = os.path.join(self.data_dir, filename)

            data_dict = torch.load(file_path, weights_only=True)

            # Get feature tensor based on specified type
            feature = data_dict[self.feature_type]

            # Choose a random melody
            melody_keys = [k for k in data_dict.keys() if k.startswith("melody_")]
            if not melody_keys:  # Skip if no melodies
                if os.path.exists(file_path):
                    os.remove(file_path)  # Remove file if no melodies
                self.file_list.remove(filename)
                print(f"Removed empty file. {filename}")
                continue

            melody_key = random.choice(melody_keys)
            tensor = data_dict[melody_key]

            # Check if the melody has any tensors
            if tensor.size(0) != 32:  # Skip if empty melody
                # wait 1 second
                import time

                time.sleep(1)
                # Remove melody from data_dict
                data_dict.pop(melody_key)
                # Save the updated data_dict
                torch.save(data_dict, file_path, _use_new_zipfile_serialization=True)
                continue

            return {"tensor": tensor, "feature": feature, "path": filename}

        raise RuntimeError("Could not find valid melody after multiple attempts")

    @staticmethod
    def get_dataloader(dataset, batch_size=64, shuffle=True, num_workers=4):
        """Creates a DataLoader for the dataset."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )


def embed_melodies(
    song_path: str,
    model: MusicVAE,
    device: str = "cpu",
    save_dir: str = "data/songs_embeddings/",
):
    """Load the tensor representation of each melody and encode it with the model into a latent space representation."""
    try:
        data_dict = torch.load(song_path, weights_only=True)
    except Exception as e:
        print(f"Error loading {song_path}: {e}")
        if os.path.exists(song_path):
            os.remove(song_path)  # Remove file if it's corrupted
        return False
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Choose a random melody
    melody_keys = [k for k in data_dict.keys() if k.startswith("melody")]
    if not melody_keys:  # Skip if no melodies
        if os.path.exists(song_path):
            os.remove(song_path)  # Remove file if no melodies
        print(f"Removed empty file. {song_path}")
        return False

    # TODO : add feature_type to model to know what features to use given the model
    feature = data_dict["OneHotGenre"].to(torch.float32).to(device)
    data_dict.pop("Features")
    old_len_data_dict = len(data_dict)
    counter = 0
    for melody_key in melody_keys:
        tensors = data_dict[melody_key]
        tensors = [t.to(torch.float32).to(device) for t in tensors]

        # Check if the melody has any tensors
        if len(tensors) == 0:
            # Remove melody from data_dict
            data_dict.pop(melody_key)
            counter += 1
            continue

        embeddings = songs_utils.chunks_to_embeddings(tensors, feature, model)
        if torch.sum(embeddings) == 0:
            print(f" !!! Embeddings are all zeros for {song_path}!!!!!")

        if device == "cpu":
            embeddings = embeddings.to(torch.float8_e4m3fn).detach()
        else:
            embeddings = embeddings.to(torch.float8_e4m3fn).detach().cpu()
        data_dict[melody_key] = embeddings

    new_len_data_dict = len(data_dict)
    if new_len_data_dict == 0 and old_len_data_dict > 0:
        print(
            f" red flag, there are melodies but all scrapped, check if they were all empty ?"
        )
        print(f" number of melodies scrapped due to empty tensor: {counter}")

    # check there are no empty melodies
    for melody_key in data_dict.keys():
        if data_dict[melody_key].size(0) == 0:
            print(f" red flag, there are empty melodies in the data_dict")
            print(f" melody_key: {melody_key}")
            print(f" data_dict[melody_key].size(0): {data_dict[melody_key].size(0)}")
            print(f" song_path: {song_path}")
            print(f" data_dict: {data_dict}")
            print(f" THIS IS file {song_path} with empty melody {melody_key}")
            print(f"--------------------------------")

    # Clean filename by removing .mid extension before adding .pt
    song_name = os.path.basename(song_path)
    save_path = os.path.join(save_dir, song_name)
    torch.save(data_dict, save_path, _use_new_zipfile_serialization=True)

    return True


def create_embedding_dataset(songs_path: str, model: MusicVAE, device: str = "cpu"):
    """Embed the tensors reprensenting the songs by melody."""
    file_names = [f for f in os.listdir(songs_path) if f.endswith(".pt")]
    file_list = [os.path.join(songs_path, f) for f in file_names]

    if device == "cpu":
        # Process MIDI files in parallel
        Parallel(n_jobs=-1)(delayed(embed_melodies)(file, model) for file in file_list)
    else:
        for file in tqdm(file_list):
            embed_melodies(file, model, device)
