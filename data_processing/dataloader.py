import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import note_seq
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Tuple
import pandas as pd
from joblib import Parallel, delayed
import random

from utils.melody_converter import melody_2bar_converter
from utils import songs_utils


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
        max_attempts = 5  # Prevent infinite loops
        for attempt in range(max_attempts):
            # Choose a random file
            filename = random.choice(self.file_list)
            file_path = os.path.join(self.data_dir, filename)

            # Check if file exists before trying to load
            if not os.path.exists(file_path):
                self.file_list.remove(filename)
                continue

            # Load the data using torch.load with error handling
            try:
                data_dict = torch.load(file_path)
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
                continue

            melody_key = random.choice(melody_keys)
            tensors = data_dict[melody_key]

            # Check if the melody has any tensors
            if len(tensors) == 0:  # Skip if empty melody
                # Remove melody from data_dict
                data_dict.pop(melody_key)
                # Save the updated data_dict
                torch.save(data_dict, file_path)
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
