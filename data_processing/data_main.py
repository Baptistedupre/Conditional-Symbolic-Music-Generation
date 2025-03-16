import os
import sys
from pathlib import Path
from preprocess import save_matching_csv
from dataloader import create_preprocessed_dataset

def check_data_requirements():
    """Check if required data files and folders exist."""
    requirements = {
        'midi_folder': Path('data/lmd_matched/lmd_matched'),
        'genre_file': Path('data/msd_tagtraum_cd1.cls'),
        'matching_file': Path('data/matching.json'),
        'output_folder': Path('data/songs')
    }
    
    # Check input requirements
    if not requirements['midi_folder'].exists():
        print("ERROR: MIDI folder not found. Please download the LMD matched dataset and extract it to data/lmd_matched/")
        print("Download from: https://colinraffel.com/projects/lmd/")
        return False
        
    if not requirements['genre_file'].exists():
        print("ERROR: Genre file not found. Please download the genre dataset and place it in data/")
        print("Download from: https://www.tagtraum.com/msd_genre_datasets.html (CD1 zip file)")
        return False
    
    # Create output directories if needed
    Path('data').mkdir(exist_ok=True)
    requirements['output_folder'].mkdir(exist_ok=True)
    
    return requirements

def count_processed_files(songs_dir):
    """Count number of processed MIDI files."""
    return len(list(Path(songs_dir).glob('*.pt')))

def main():
    """Main function to run the data preprocessing pipeline."""
    print("Checking data requirements...")
    requirements = check_data_requirements()
    if not requirements:
        sys.exit(1)
    
    # Create matching.json if it doesn't exist or is empty
    if not Path('data/matching.json').exists() or Path('data/matching.json').stat().st_size == 0:
        print("Creating matching.json...")
        save_matching_csv(
            'data/msd_tagtraum_cd1.cls',
            'data/lmd_matched/lmd_matched',
            'data/matching.json'
        )
    else:
        print("matching.json already exists, skipping creation")
    
    # Check existing processed files
    n_processed = count_processed_files('data/songs')
    if n_processed > 0:
        print(f"Found {n_processed} already processed files in data/songs/")
        user_input = input("Do you want to process remaining files? [y/N]: ")
        if user_input.lower() != 'y':
            print("Skipping preprocessing")
            return
    
    # Process MIDI files
    print("Processing MIDI files...")
    create_preprocessed_dataset('data/matching.json')
    
    # Final count
    final_count = count_processed_files('data/songs')
    print(f"Preprocessing complete. {final_count} files processed in total.")

if __name__ == "__main__":
    main()
