from typing import Optional, Union, List, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
ArrayLike = Tuple[np.ndarray, list]

tokens_protein = "ACDEFGHIKLMNPQRSTVWY-"
tokens_rna = "ACGU-"
tokens_dna = "ACGT-"

def get_tokens(alphabet):
    assert isinstance(alphabet, str), "Argument 'alphabet' must be of type str"
    if alphabet == "protein":
        return tokens_protein
    elif alphabet == "rna":
        return tokens_rna
    elif alphabet == "dna":
        return tokens_dna
    else:
        return alphabet
    
def validate_alphabet(sequences : ArrayLike, tokens : str):
    all_char = "".join(sequences)
    tokens_data = "".join(sorted(set(all_char)))
    sorted_tokens = "".join(sorted(tokens))
    if not sorted_tokens == tokens_data:
        raise KeyError(f"The chosen alphabet is incompatible with the Multi-Sequence Alignment. The missing tokens are: {[c for c in tokens_data if c not in sorted_tokens]}. Consider using another alphabet using the 'alphabet' argument.")
    
def encode_sequence(sequence : str, tokens : str) -> list:
    """Takes a string sequence in input an returns the numeric encoding.

    Args:
        sequence (str): Input sequence.
        tokens (str): Alphabet to be used for the encoding.

    Returns:
        list: Encoded sequence.
    """
    letter_map = {l : n for n, l in enumerate(tokens)}
    return np.array([letter_map[l] for l in sequence])

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
    def __init__(self,
                 data_path : Union[str, Path],
                 ann_path : Union[str, Path]=None,
                 colors_path : Optional[Union[str, Path]]=None,
                 alphabet : str="protein"):
        """Initialize the dataset.

        Args:
            data_path (Union[str, Path]): Path to the data file (plain text or fasta).
            ann_path (Union[str, Path], optional): Path to the annotations file (csv). Defaults to None.
            colors_path (Union[str, Path], optional): Path to the color mapping file (csv). If None, colors are assigned automatically. Defaults to None.
            alphabet (str, optional): Selects the type of encoding of the sequences. Default choices are ("protein", "rna", "dna"). Defaults to "protein".
        """
        self.names = []
        self.data = []
        self.labels = []
        self.colors = []
        self.tokens = None # Only needed for protein sequence data
        
        # Automatically detects if the file is in fasta format and imports the data
        with open(data_path, "r") as f:
            first_line = f.readline()
        if first_line.startswith(">"):
            # Select the proper encoding
            self.tokens = get_tokens(alphabet)
            names, sequences = import_from_fasta(data_path)
            validate_alphabet(sequences=sequences, tokens=self.tokens)
            self.names = np.array(names).astype(str)
            self.data = np.vectorize(encode_sequence, excluded=["tokens"], signature="(), () -> (n)")(sequences, self.tokens)
        else:
            with open(data_path, "r") as f:
                for line in f:
                    self.data.append(line.strip().split())
            self.data = np.array(self.data, dtype=np.float32)
            self.names = np.arange(len(self.data)).astype("str")
        
        # Load annotations
        if ann_path:
            ann_df = pd.read_csv(ann_path).astype(str)
            self.legend = [n for n in ann_df.columns if n != "Name"]

            # Validate the legend format: special characters are not allowed
            special_characters = '!@#$%^&*()-+?=,<>/'
            for leg in self.legend:
                if any(c in special_characters for c in leg):
                    raise KeyError("Legend names can't contain any special characters.")

            for leg in self.legend:
                self.labels.append({str(n) : str(l) for n, l in zip(ann_df["Name"], ann_df[leg])})
        else:
            self.legend = None
            self.labels = None
        
        # Load colors
        if colors_path is not None:
            df_colors = pd.read_csv(colors_path)
            for leg in self.legend:
                df_leg = df_colors.loc[df_colors["Legend"] == leg]
                self.colors.append({str(n) : c for n, c in zip(df_leg["Label"], df_leg["Color"])})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx : int) -> Any:
        sample = self.data[idx]
        return sample
    
    def get_num_visibles(self) -> int:
        return self.data.shape[1]
    
    def get_num_states(self) -> int:
        return np.max(self.data) + 1