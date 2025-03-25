import numpy as np
import pandas as pd
import torch
import note_seq 
import tempfile
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_processing import preprocess
from math import erf, sqrt

from MusicVAE.models.diffusion_transformer.denoiser_transformer import sample_fn
from utils.songs_utils import embeddings_to_song
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils.songs_utils import one_hot_along_dim1, Song
from utils.melody_converter import melody_2bar_converter 


# Function 1: Build a cluster-based classifier from matching.json.
def train_supervised_classifier(matching_path, random_state=42):
    """
    Reads matching_modified.json and trains a supervised classifier on the 'Features'
    to predict the 'Genre'. Trains on the full dataset (no test/train split),
    then returns a classifier function that maps a feature vector to its predicted genre,
    along with the trained classifier and its training accuracy.

    Args:
      matching_path: Path to the modified matching JSON file.
      random_state: Random seed for reproducibility.

    Returns:
      classifier_fn: A function that takes a feature vector and returns a predicted genre.
      clf: The trained classifier model.
    """
    matching = pd.read_json(matching_path)
    X = np.array(matching["Features"].tolist())
    y = np.array(matching["Genre"].tolist())

    # Use only the Random Forest classifier and train on all data
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print("Supervised Training Accuracy on full dataset:", accuracy)

    def classifier_fn(feature_vec):
        feature_vec = np.array(feature_vec).reshape(1, -1)
        return clf.predict(feature_vec)[0]

    return classifier_fn, clf


# Function 2: Sample songs from DDPM and decode with VAE.
def sample_genres(
    vae, ddpm, matching_path, device, feature_dim, seq_len, latent_dim, seed
):
    """
    Generates latent samples using ddpm, decodes them with vae, and returns a dictionary mapping
    each genre to its corresponding [Song, decoded_tensors] (using one-hot identity mapping from matching.json).
    """
    songs_dict = {}
    matching = pd.read_json(matching_path)
    matching["OneHotGenreTuple"] = matching["OneHotGenre"].apply(lambda x: tuple(x))
    feature_mapping = {}
    for f in np.eye(feature_dim):
        f_tuple = tuple(f.tolist())
        matching_row = matching[matching["OneHotGenreTuple"] == f_tuple]
        if not matching_row.empty:
            genre_name = matching_row["Genre"].iloc[0]
            feature_mapping[f_tuple] = genre_name
        else:
            print(f"Warning: no matching genre found for feature {f_tuple}")
    sigma_begin = 1.0
    sigma_end = 0.01
    num_sampling_steps = 1000
    sampling_sigmas = np.geomspace(
        sigma_begin, sigma_end, num=num_sampling_steps
    ).tolist()
    features = torch.eye(feature_dim).to(device)
    latent_samples = sample_fn(
        ddpm,
        sampling_sigmas,
        device,
        "ddpm",
        features,
        num_samples=feature_dim,
        sample_shape=(seq_len, latent_dim),
        seed=seed,
    )
    for i, latent in enumerate(latent_samples):
        feature = features[i]
        feature_tuple = tuple(feature.detach().cpu().numpy().tolist())
        song, _ = embeddings_to_song(latent, feature, vae)
        genre = feature_mapping.get(feature_tuple, "Unknown")
        songs_dict[genre] = song
    return songs_dict


def compute_song_features_from_midi(song):
    """
    Alternative to compute_song_features:
    Downloads the Song object as a MIDI file to a temporary location,
    extracts features using preprocess.get_features, then removes the temporary file.
    Assumes the Song object has a .download(filepath) method.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            midi_path = tmp.name
        song.download(midi_path)  # Song expected to write MIDI content to midi_path
        features = preprocess.get_features(midi_path)
    except Exception as e:
        features = [0.0, 0.0, 0.0, 0.0, 0.0]
    finally:
        if os.path.exists(midi_path):
            os.remove(midi_path)
    return features


# Function 3: Check clustering on generated samples.
def check_clusters(songs_dict_list, classifier_fn):
    """
    For each generated sample (in a list) for every genre in songs_dict_list, extract features with compute_song_features(),
    predict its genre using classifier_fn, and compute overall and per-genre accuracy.

    songs_dict_list: dict where keys are genres and values are lists of (song, decoded_tensors)
    """
    total = 0
    correct = 0
    genre_stats = {}
    for genre, sample_list in songs_dict_list.items():
        for song in sample_list:
            feat = compute_song_features_from_midi(song)
            pred = classifier_fn(feat)
            is_corr = pred == genre
            total += 1
            if is_corr:
                correct += 1
            if genre not in genre_stats:
                genre_stats[genre] = {"total": 0, "correct": 0}
            genre_stats[genre]["total"] += 1
            if is_corr:
                genre_stats[genre]["correct"] += 1
    overall_accuracy = correct / total if total > 0 else 0.0
    per_genre_accuracy = {
        g: (d["correct"] / d["total"]) for g, d in genre_stats.items() if d["total"] > 0
    }
    return {
        "overall_accuracy": overall_accuracy,
        "per_genre_accuracy": per_genre_accuracy,
    }


def compute_frame_statistics(song, win_measures=4, hop_measures=2):
    """
    Splits song.note_sequence into measures (assuming 4/4 time with measure duration = 4 beats).
    Computes frame statistics (mean and variance of pitch and note duration) for each sliding window.
    """
    ns = song.note_sequence
    # Estimate tempo using qpm (beats per minute)
    tempo = ns.tempos[0].qpm if ns.tempos and hasattr(ns.tempos[0], "qpm") else 120.0
    # Assume 4/4 time; measure_duration in seconds = (4 beats)*(60/tempo)
    measure_duration = 4 * (60.0 / tempo)
    # Group notes into measures based on their start time
    measures = {}
    for note in ns.notes:
        key = int(note.start_time // measure_duration)
        measures.setdefault(key, []).append(note)
    # For each measure that has notes, compute mean pitch and mean duration
    measure_stats = []
    sorted_keys = sorted(measures.keys())
    for key in sorted_keys:
        notes = measures[key]
        if not notes:
            continue
        pitches = [n.pitch for n in notes]
        durations = [n.end_time - n.start_time for n in notes]
        measure_stats.append(
            {
                "pitch_mean": np.mean(pitches),
                "pitch_var": np.var(pitches) if len(pitches) > 1 else 0.0,
                "dur_mean": np.mean(durations),
                "dur_var": np.var(durations) if len(durations) > 1 else 0.0,
            }
        )
    # Slide a window over measures
    frames = []
    for start in range(0, len(measure_stats) - win_measures + 1, hop_measures):
        window = measure_stats[start : start + win_measures]
        # Aggregate all pitches and durations in the window
        all_pitches = []
        all_durs = []
        for stats in window:
            # For simplicity, use the measure mean values repeated once per measure.
            # (Alternatively, aggregate every note's value.)
            all_pitches.append(stats["pitch_mean"])
            all_durs.append(stats["dur_mean"])
        frame_stats = {
            "pitch_mu": np.mean(all_pitches),
            "pitch_var": np.var(all_pitches) if len(all_pitches) > 1 else 0.0,
            "dur_mu": np.mean(all_durs),
            "dur_var": np.var(all_durs) if len(all_durs) > 1 else 0.0,
        }
        frames.append(frame_stats)
    return frames


def overlapping_area(gauss1, gauss2):
    """
    Computes the overlapping area (OA) between two Gaussian PDFs.
    gauss1 and gauss2 are tuples: (mu, sigma), where sigma > 0.
    If sigma==0, returns 1 if means are equal, else 0.
    """
    mu1, sigma1 = gauss1
    mu2, sigma2 = gauss2
    # Ensure mu1 <= mu2
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
        sigma1, sigma2 = sigma2, sigma1
    # If sigma nearly zero, treat as Dirac delta
    if sigma1 < 1e-6 or sigma2 < 1e-6:
        return 1.0 if abs(mu1 - mu2) < 1e-3 else 0.0
    # Compute intersection point c by solving: pdf1(c)=pdf2(c)
    # When variances differ, the closed form is:
    try:
        a = 1 / (2 * sigma1**2) - 1 / (2 * sigma2**2)
        b = mu2 / (sigma2**2) - mu1 / (sigma1**2)
        c = -b / (2 * a)
    except ZeroDivisionError:
        c = (mu1 + mu2) / 2
    # Compute OA
    term1 = erf((c - mu1) / (sqrt(2) * sigma1))
    term2 = erf((c - mu2) / (sqrt(2) * sigma2))
    OA = 1 - term1 + term2
    # Clip OA to [0,1]
    return max(0, min(OA, 1))


def compute_OA_metrics_for_song(song):
    """
    For a given Song, compute OA values for pitch and duration between each adjacent frame.
    Returns two lists: list of OA_pitch and OA_duration values.
    """
    frames = compute_frame_statistics(song)
    oa_pitch = []
    oa_dur = []
    # Compute OA for adjacent frame pairs
    for k in range(len(frames) - 1):
        f1 = frames[k]
        f2 = frames[k + 1]
        # For pitch
        mu1, var1 = f1["pitch_mu"], f1["pitch_var"]
        mu2, var2 = f2["pitch_mu"], f2["pitch_var"]
        sigma1 = sqrt(var1) if var1 > 0 else 1e-6
        sigma2 = sqrt(var2) if var2 > 0 else 1e-6
        oa_p = overlapping_area((mu1, sigma1), (mu2, sigma2))
        oa_pitch.append(oa_p)
        # For duration
        mu1, var1 = f1["dur_mu"], f1["dur_var"]
        mu2, var2 = f2["dur_mu"], f2["dur_var"]
        sigma1 = sqrt(var1) if var1 > 0 else 1e-6
        sigma2 = sqrt(var2) if var2 > 0 else 1e-6
        oa_d = overlapping_area((mu1, sigma1), (mu2, sigma2))
        oa_dur.append(oa_d)
    return oa_pitch, oa_dur


def evaluate_OA_similarity(songs_dict_list, gt_pitch_stats, gt_dur_stats):
    """
    Given a dict of generated songs (keys=genre, values=list of Song objects),
    and ground-truth statistics for OA (tuple: (mu_GT, var_GT)) for pitch and duration,
    compute the relative similarity metrics.

    For each set of OA values for a song, we compute:
      Consistency = max(0, 1 - |μ_OA - μ_GT| / μ_GT)
      Variance = max(0, 1 - |σ²_OA - σ²_GT| / σ²_GT)
    We then average these values across all songs (overall and per genre).

    Returns a dict:
      {
         "overall": {"pitch_consistency": ..., "pitch_variance": ..., "dur_consistency": ..., "dur_variance": ...},
         "per_genre": {genre: { ... }, ...}
      }

    Any error above 100% is clipped to 0.
    """
    overall_pitch_consistency = []
    overall_pitch_variance = []
    overall_dur_consistency = []
    overall_dur_variance = []

    per_genre = {}

    # Process each genre's generated songs.
    for genre, song_list in songs_dict_list.items():
        genre_pitch_cons = []
        genre_pitch_var = []
        genre_dur_cons = []
        genre_dur_var = []
        for song in song_list:
            oa_pitch, oa_dur = compute_OA_metrics_for_song(song)
            if not oa_pitch or not oa_dur:
                continue
            mu_oa_pitch = np.mean(oa_pitch)
            var_oa_pitch = np.var(oa_pitch)
            mu_oa_dur = np.mean(oa_dur)
            var_oa_dur = np.var(oa_dur)
            # Compute relative similarity (clip error >=100% to 0)
            p_cons = (
                max(0, 1 - abs(mu_oa_pitch - gt_pitch_stats[0]) / gt_pitch_stats[0])
                if gt_pitch_stats[0] > 0
                else 0
            )
            p_var = (
                max(0, 1 - abs(var_oa_pitch - gt_pitch_stats[1]) / gt_pitch_stats[1])
                if gt_pitch_stats[1] > 0
                else 0
            )
            d_cons = (
                max(0, 1 - abs(mu_oa_dur - gt_dur_stats[0]) / gt_dur_stats[0])
                if gt_dur_stats[0] > 0
                else 0
            )
            d_var = (
                max(0, 1 - abs(var_oa_dur - gt_dur_stats[1]) / gt_dur_stats[1])
                if gt_dur_stats[1] > 0
                else 0
            )

            overall_pitch_consistency.append(p_cons)
            overall_pitch_variance.append(p_var)
            overall_dur_consistency.append(d_cons)
            overall_dur_variance.append(d_var)

            genre_pitch_cons.append(p_cons)
            genre_pitch_var.append(p_var)
            genre_dur_cons.append(d_cons)
            genre_dur_var.append(d_var)

        if genre_pitch_cons:
            per_genre[genre] = {
                "pitch_consistency": np.mean(genre_pitch_cons),
                "pitch_variance": np.mean(genre_pitch_var),
                "dur_consistency": np.mean(genre_dur_cons),
                "dur_variance": np.mean(genre_dur_var),
            }
    overall = {
        "pitch_consistency": (
            np.mean(overall_pitch_consistency) if overall_pitch_consistency else 0.0
        ),
        "pitch_variance": (
            np.mean(overall_pitch_variance) if overall_pitch_variance else 0.0
        ),
        "dur_consistency": (
            np.mean(overall_dur_consistency) if overall_dur_consistency else 0.0
        ),
        "dur_variance": np.mean(overall_dur_variance) if overall_dur_variance else 0.0,
    }
    return {"overall": overall, "per_genre": per_genre}


# New function: generate_and_evaluate_fixed that loops exactly n_songs_per_genre times.
def generate_and_evaluate(
    vae,
    ddpm,
    matching_path,
    device,
    feature_dim,
    seq_len,
    latent_dim,
    n_songs_per_genre,
    gt_pitch_stats=(59.45755364275701, 7.664131456564755),
    gt_dur_stats=(0.5983641122797528, 0.10269438441810701),
):
    """
    Generates samples from ddpm+vae for n_songs_per_genre iterations. For each iteration, sample_genres()
    returns one sample per genre. Samples are aggregated into a dict mapping each genre to a list of
    n_songs_per_genre Song objects. Then two evaluations are performed:
      1. Clustering accuracy, using compute_song_features_from_midi and classifier from matching.json.
      2. OA similarity, computing consistency and variance metrics for pitch and duration.

    Returns:
      dict: {
              "cluster_metrics": {"overall_accuracy": ..., "per_genre_accuracy": ...},
              "OA_metrics": {
                  "overall": {"pitch_consistency": ..., "pitch_variance": ..., "dur_consistency": ..., "dur_variance": ...},
                  "per_genre": {genre: { ... }, ...}
              }
            }
    """
    collected = {}  # genre -> list of Song objects
    for i in range(n_songs_per_genre):
        new_samples = sample_genres(
            vae, ddpm, matching_path, device, feature_dim, seq_len, latent_dim, seed=i
        )
        for genre, song in new_samples.items():
            collected.setdefault(genre, []).append(song)

    # Get classifier from matching.json and compute clustering accuracy.
    classifier_fn, _ = train_supervised_classifier(matching_path)
    cluster_res = check_clusters(collected, classifier_fn)

    # Compute OA similarity metrics.
    oa_res = evaluate_OA_similarity(collected, gt_pitch_stats, gt_dur_stats)

    return {"cluster_metrics": cluster_res, "OA_metrics": oa_res}


# Placeholder helper: implement your own melody loader from a MIDI file.
def load_melody(midi_path):
    name = os.path.splitext(os.path.basename(midi_path))[0]
    file_path = f"data/songs/{name}.pt"
    data_dict = torch.load(file_path)
    melody_keys = [k for k in data_dict.keys() if "melody" in k]
    for melody_key in melody_keys:
        if data_dict[melody_key].shape[0] != 0:
            melody = data_dict[melody_key].to(torch.float32)
            return melody


def reconstruct_and_evaluate(
    vae,
    matching_path,
    device,
    n_songs_per_genre,
    gt_pitch_stats=(59.45755364275701, 7.664131456564755),
    gt_dur_stats=(0.5983641122797528, 0.10269438441810701),
):
    """
    For each genre in matching_path, selects n_songs_per_genre songs,
    reconstructs them via the VAE (using one_hot_along_dim1 and melody_2bar_converter),
    and collects the reconstructed Song objects in a dict.
    Then, obtains classifier from matching.json to compute clustering and OA metrics.
    """
    # Read matching DataFrame.
    df = pd.read_json(matching_path)
    collected = {}

    # Process each genre.
    for genre, group in df.groupby("Genre"):
        collected[genre] = []
        selected = group.head(n_songs_per_genre)
        for _, row in selected.iterrows():
            midi_path = row["Path"]
            try:
                # Load the melody (user to implement load_melody)
                melody = load_melody(
                    midi_path
                )  # returns a tensor or iterable of chunks
            except Exception as e:
                print(f"Skipping {midi_path}: {e}")
                continue

            # Get one-hot genre feature and send to device.
            feat = torch.tensor(row["OneHotGenre"], dtype=torch.float32).to(device)
            reconstructed_chunks = []
            for chunk in melody:
                recon, _, _, _ = vae(chunk.to(device).unsqueeze(0), feat.unsqueeze(0))
                decoded_tensor = one_hot_along_dim1(recon.squeeze(0).detach())
                reconstructed_chunks.append(decoded_tensor.unsqueeze(0).cpu())
            if not reconstructed_chunks:
                continue
            reconstructed = torch.cat(reconstructed_chunks, dim=0)
            recon_chunks = melody_2bar_converter.from_tensors(reconstructed)
            concat_chunks = note_seq.sequences_lib.concatenate_sequences(recon_chunks)
            # Create a reconstructed Song instance.
            s_recon = Song(
                concat_chunks,
                melody_2bar_converter,
                chunk_length=2,
                multitrack=False,
                reconstructed=True,
            )
            collected[genre].append(s_recon)

    # Get classifier from matching.json and compute clustering accuracy.
    classifier_fn, _ = train_supervised_classifier(matching_path)
    cluster_res = check_clusters(collected, classifier_fn)

    # Compute OA similarity metrics.
    oa_res = evaluate_OA_similarity(collected, gt_pitch_stats, gt_dur_stats)

    return {"cluster_metrics": cluster_res, "OA_metrics": oa_res}


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    import gc
    import pickle
    import torch
    from models.music_vae.model import MusicVAE

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae_weights = [
        (i, f"output/model_epoch_{i}.pt") 
        for i in range(50) 
        if i % 2 == 0 and os.path.exists(f"output/model_epoch_{i}.pt")
    ]
    progression_dict = {}
    for (i, path) in vae_weights:
        print(f"Loading VAE weights from {path}")
        model = MusicVAE(input_size=90, output_size=90, latent_dim=512, device=device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(device)
        results = reconstruct_and_evaluate(
            model,
            "data/matching.json",
            device,
            n_songs_per_genre=10,
        )
        loss = checkpoint['average_loss']
        overall_accuracy = results["cluster_metrics"]["overall_accuracy"]
        overall_pitch_consistency = results["OA_metrics"]["overall"]["pitch_consistency"]
        overall_pitch_variance = results["OA_metrics"]["overall"]["pitch_variance"]
        overall_dur_consistency = results["OA_metrics"]["overall"]["dur_consistency"]
        overall_dur_variance = results["OA_metrics"]["overall"]["dur_variance"]
        progression_dict[i] = {
            "loss": loss,
            "overall_accuracy": overall_accuracy,
            "pitch_consistency": overall_pitch_consistency,
            "pitch_variance": overall_pitch_variance,
            "dur_consistency": overall_dur_consistency,
            "dur_variance": overall_dur_variance,
        }

        with open("output/progression_dict.pkl", "wb") as f:
            pickle.dump(progression_dict, f)
            f.close()

        with open(f"output/results_dict_epoch_{i}.pkl", "wb") as f:
            pickle.dump(results, f)
            f.close()

        # Empty torch cache
        torch.cuda.empty_cache()
        gc.collect()
        del model