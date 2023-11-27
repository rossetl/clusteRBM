from typing import Optional, Union, List, Any
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader

def sequence_to_numeric(string : str) -> list:
    amino_letters = 'ACDEFGHIKLMNPQRSTVWY-'
    letter_map = {l : n for l, n in zip(amino_letters, range(21))}
    n_list = []
    for l in string:
        n_list.append(letter_map[l])
    return n_list

def import_from_fasta(fasta_name : Union[str, Path]) -> List[list]:
    sequences = []
    names = []
    seq = ''
    with open(fasta_name, 'r') as f:
        first_line = f.readline()
        if not first_line.startswith('>'):
            raise RuntimeError(f"The file {fasta_name} is not in a fasta format.")
        f.seek(0)
        for line in f:
            if not line.strip():
                continue
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                header = line[1:].strip().replace(' ', '_')
                names.append(header)
                seq = ''
            else:
                seq += line.strip()
    if seq:
        sequences.append(seq)
    
    return names, sequences

class DatasetRBM(Dataset):
    def __init__(self, data_path : Union[str, Path], ann_path : Union[str, Path]):
        """Initialize the dataset.

        Args:
            data_path (Union[str, Path]): Path to the data file (plain text or fasta).
            ann_path (Union[str, Path]): Path to the annotations file (csv).
        """
        self.names = []
        self.data = []
        self.labels = []
        
        # Automatically detects if the file is in fasta format and imports the data
        with open(data_path, "r") as f:
            first_line = f.readline()
        if first_line.startswith(">"):
            names, sequences = import_from_fasta(data_path)
            self.names = np.array(names)
            self.data = np.array(list(map(sequence_to_numeric, sequences)), dtype=np.int64)
        else:
            with open(data_path, "r") as f:
                for line in f:
                    self.data.append(line.strip().split())
            self.data = np.array(self.data, dtype=np.float32)
            self.names = np.arange(len(self.data)).astype(str)
        
        with open(ann_path, "r") as f:
            headers = f.readline().strip().split(",")
            n_class = len(headers) - 1
            for _ in range(n_class):
                self.labels.append({n : "-1" for n in self.names})
            for line in f:
                n, *l = line.strip().split(",")
                for i, lab_class in enumerate(self.labels):
                    lab_class[n] = list(l)[i]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx : int) -> Any:
        sample = self.data[idx]
        return sample
    
    def get_num_visibles(self) -> int:
        return self.data.shape[1]
    
    def get_num_states(self) -> int:
        return np.max(self.data) + 1