# MusicVAE
PyTorch implementation of Magenta's MusicVAE

## Creating the Data for MIDIDataset

### 1. Download and Extract Data

Before using the `MIDIDataset` class, you'll need to download and properly extract two main datasets:

- **Genre Label File (.cls):**  
  Download the genre label file from the Tagtraum website. For instance, you can get it from the [Genre Ground truth (CD1)](https://www.tagtraum.com/msd_genre_datasets.html) page. 
  Scroll down to the Genre Ground Truth section to download the zip file under the label "CD1". 
  **Extract:** Place the extracted `.cls` file in the `data/` directory.

- **LMD Matched Dataset:**  
  Download the LMD Matched dataset from Colin Raffel’s website, available at [LMD Matched](https://colinraffel.com/projects/lmd/).  
  Scroll down and click on the link in the section that says "LMD-matched" to download a midi dataset where each file is matched to an entry in the million song dataset.
  **Extract:** Make sure the extracted dataset is located in `data/lmd_matched/lmd_matched/`.

### 2. Preprocess the Data

Once the data is in place, run the preprocessing script to prepare the data for the `MIDIDataset` class. Open your terminal and execute:

```bash
python data_main.py
```

### 3. Use MIDIDataset
```python
from data_processing.dataloader import MIDIDataset

dataset = MIDIDataset("data/songs/", feature_type="Features")
dataloader = MIDIDataset.get_dataloader(dataset, batch_size=64)
```