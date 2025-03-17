import sys
import os
import argparse
from pathlib import Path
import s3fs
from preprocess import save_matching_csv
from dataloader import create_preprocessed_dataset


def check_data_requirements(local: bool, fs=None):
    """Check if required data files and folders exist."""
    if local:
        requirements = {
            "midi_folder": Path("data/lmd_matched/lmd_matched"),
            "genre_file": Path("data/msd_tagtraum_cd1.cls"),
            "matching_file": Path("data/matching.json"),
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
        Path("data").mkdir(exist_ok=True)
        requirements["output_folder"].mkdir(exist_ok=True)
        return requirements
    else:
        base = "s3://lstepien/Conditional_Music_Generation/data/"
        requirements = {
            "midi_folder": base + "lmd_matched",
            "genre_file": base + "msd_tagtraum_cd1.cls",
            "matching_file": base + "matching.json",
            "output_folder": base + "songs",
        }
        if not fs.exists(requirements["midi_folder"]):
            print("ERROR: Remote MIDI folder not found. Please check S3 storage.")
            return False
        if not fs.exists(requirements["genre_file"]):
            print("ERROR: Remote Genre file not found. Please check S3 storage.")
            return False
        if not fs.exists(requirements["output_folder"]):
            fs.mkdir(requirements["output_folder"])
        return requirements


def count_processed_files(songs_dir, local: bool, fs=None):
    """Count number of processed MIDI files."""
    if local:
        return len(list(Path(songs_dir).glob("*.pt")))
    else:
        return len(fs.glob(songs_dir + "/*.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-local", action="store_true", help="Run in local mode (assumes local data)"
    )
    args = parser.parse_args()

    if not args.local:
        fs = s3fs.S3FileSystem(
            client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
        )
    else:
        fs = None

    print("Checking data requirements...")
    reqs = check_data_requirements(args.local, fs)
    if not reqs:
        sys.exit(1)

    # Use matching.json path based on mode.
    matching_path = reqs["matching_file"] if not args.local else "data/matching.json"

    if args.local:
        if not Path(matching_path).exists() or Path(matching_path).stat().st_size == 0:
            print("Creating matching.json...")
            save_matching_csv(
                "data/msd_tagtraum_cd1.cls",
                "data/lmd_matched/lmd_matched",
                matching_path,
            )
        else:
            print("matching.json already exists, skipping creation")
    else:
        if not fs.exists(matching_path) or fs.info(matching_path)["Size"] == 0:
            print("Creating matching.json on S3...")
            local_temp = "temp_matching.json"
            save_matching_csv(
                "s3://lstepien/Conditional_Music_Generation/data/msd_tagtraum_cd1.cls",
                "s3://lstepien/Conditional_Music_Generation/data/lmd_matched",
                local_temp,
            )
            fs.put(local_temp, matching_path)
            os.remove(local_temp)
        else:
            print("matching.json already exists on S3, skipping creation")

    # Check processed files count.
    songs_path = "data/songs" if args.local else reqs["output_folder"]
    n_processed = count_processed_files(songs_path, args.local, fs)
    if n_processed > 0:
        print(f"Found {n_processed} already processed files.")
        user_input = input("Do you want to process remaining files? [y/N]: ")
        if user_input.lower() != "y":
            print("Skipping preprocessing")
            return

    print("Processing MIDI files...")
    create_preprocessed_dataset(matching_path)
    final_count = count_processed_files(songs_path, args.local, fs)
    print(f"Preprocessing complete. {final_count} files processed in total.")


if __name__ == "__main__":
    main()
