import sys
import os
import shutil  # For unzipping
from pathlib import Path
import gdown  # pip install gdown
from preprocess import save_matching_csv
from dataloader import create_preprocessed_dataset

def main():
    # Google Drive URLs for the two files.
    local_matching = "data/matching.json"
    # Updated matching.json file ID from provided URL.
    remote_matching_url = "https://drive.google.com/uc?id=1ouJuM2FW0XpTNelE6OD9_GMvaMI2vg2c&export=download"
    songs_path = Path("data/songs")
    # Updated songs.zip file ID from provided URL.
    remote_songs_zip_url = "https://drive.google.com/uc?id=1kG9nEM3p2qv4rQ9AiW-4U3i5ba8qkVHF&export=download"
    
    # Attempt to download matching.json if not present locally.
    if not Path(local_matching).exists():
        print("Found no local matching.json. Downloading from Google Drive...")
        os.makedirs("data", exist_ok=True)
        gdown.download(remote_matching_url, local_matching, quiet=False)
    else:
        print("Local matching.json found.")
        
    # Attempt to download songs.zip if processed files are missing.
    processed_files = list(songs_path.glob("*.pt")) if songs_path.exists() else []
    if not processed_files:
        print("No processed song files found.")
        print("Downloading songs.zip from Google Drive and unzipping...")
        local_zip = "data/songs.zip"
        os.makedirs("data", exist_ok=True)
        gdown.download(remote_songs_zip_url, local_zip, quiet=False)
        shutil.unpack_archive(local_zip, "data/")
        os.remove(local_zip)
    else:
        print("Found processed song files locally.")

    # Check if both matching.json and processed songs are available.
    processed_files = list(songs_path.glob("*.pt")) if songs_path.exists() else []
    if Path(local_matching).exists() and processed_files:
        print("Local matching.json and processed songs are available. Data is ready for MIDIDataset.")
        return

    # Proceed with local preprocessing.
    print("Checking local data requirements...")
    reqs = check_data_requirements()
    if not reqs:
        sys.exit(1)

    if not Path(local_matching).exists() or Path(local_matching).stat().st_size == 0:
        print("Creating matching.json locally...")
        save_matching_csv("data/msd_tagtraum_cd1.cls", "data/lmd_matched/lmd_matched", local_matching)
    else:
        print("matching.json is available.")

    processed_files = list(songs_path.glob("*.pt"))
    if processed_files:
        print(f"Found {len(processed_files)} processed files.")
        user_input = input("Do you want to process remaining files? [y/N]: ")
        if user_input.lower() != "y":
            print("Skipping preprocessing")
            return
    else:
        print("No processed files found. Proceeding with preprocessing...")

    print("Processing MIDI files...")
    create_preprocessed_dataset(local_matching)
    final_count = len(list(songs_path.glob("*.pt")))
    print(f"Preprocessing complete. {final_count} files processed in total.")

    # Embedding dataset creation step
    songs_emb_path = Path("data/songs_embeddings")
    processed_emb_files = (
        list(songs_emb_path.glob("*.pt")) if songs_emb_path.exists() else []
    )
    if not processed_emb_files:
        remote_embeddings_zip = (
            "s3://lstepien/Conditional_Music_Generation/data/songs_embeddings.zip"
        )
        if fs.exists(remote_embeddings_zip):
            print("Found remote songs_embeddings.zip. Downloading and unzipping...")
            local_zip = "data/songs_embeddings.zip"
            fs.get(remote_embeddings_zip, local_zip)
            shutil.unpack_archive(local_zip, "data/")
            os.remove(local_zip)
        elif list(songs_path.glob("*.pt")) and Path("output/model_vae.pt").exists():
            print("Creating embeddings from processed songs...")
            import torch
            from models.music_vae.model import MusicVAE
            from dataloader import create_embedding_dataset

            input_dim = 90
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = MusicVAE(
                input_size=input_dim,
                output_size=input_dim,
                latent_dim=512,
                device=device,
            )  # noqa 501
            model.to(device)
            checkpoint = torch.load("output/model_vae.pt", map_location=device)
            state_dict = checkpoint["model_state_dict"]
            model.load_state_dict(state_dict)
            model.eval()
            create_embedding_dataset("data/songs/", model, device)
        else:
            print("No source for creating embeddings found.")
    else:
        print("Processed embeddings are available.")


# Include check_data_requirements unchanged
def check_data_requirements():
    requirements = {
        "midi_folder": Path("data/lmd_matched/lmd_matched"),
        "genre_file": Path("data/msd_tagtraum_cd1.cls"),
        "output_folder": Path("data/songs"),
    }
    if not requirements["midi_folder"].exists():
        print(
            "ERROR: MIDI folder not found. Please download the LMD matched dataset and extract it to data/lmd_matched/"
        )
        print("Download from: https://colinraffel.com/projects/lmd/")
        return False
    if not requirements["genre_file"].exists():
        print(
            "ERROR: Genre file not found. Please download the genre dataset and place it in data/"
        )
        print(
            "Download from: https://www.tagtraum.com/msd_genre_datasets.html (CD1 zip file)"
        )
        return False
    os.makedirs("data", exist_ok=True)
    requirements["output_folder"].mkdir(exist_ok=True)
    return requirements

if __name__ == "__main__":
    main()