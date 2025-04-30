# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
# Adapted from https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
from pathlib import Path
from typing import Any, List, Union
from torch.utils.data.dataloader import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import Dataset

from src.dataloaders.base import SequenceDataset, default_data_path
# genomics datasets
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.dataloaders.datasets.gene_expression import EnformerDataset
from src.dataloaders.genomics import HG38

"""

Dataloaders for genomics datasets, including pretraining and downstream tasks.  First works in HyenaDNA project, May 2023.

"""

class EnformerDataloader(HG38):
    _name_ = "gene_expression"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False, 
                detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                fast_forward_epochs=None, fast_forward_batches=None, organism='human', seq_len=196608, l_output=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.l_output = l_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.organism = organism
        self.seq_len = seq_len

        if self.dest_path is None:
            self.dest_path = default_data_path / "Enformer"

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
        
        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            EnformerDataset(split=split,
                                max_length=max_len,
                                seq_len=self.seq_len,
                                dataset_name=self.dataset_name,
                                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                                tokenizer_name=self.tokenizer_name,
                                use_padding=self.use_padding,
                                d_output=self.d_output,
                                add_eos=self.add_eos,
                                dest_path=self.dest_path,
                                organism=self.organism,
                                return_mask=self.return_mask,
            )
            for split, max_len in zip(['train', 'val', 'test'], [self.max_length, self.max_length_val, self.max_length_test])
        ]
    
    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return self._data_loader(self.dataset_train, batch_size=self.batch_size)

    def _data_loader(self, dataset: EnformerDataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=self.pin_memory, drop_last=self.drop_last, sampler=sampler)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)
    


if __name__ == '__main__':
    """Quick test using dataloader. Can't call from here though.
        python -m src.dataloaders.enformer
    """

    loader = EnformerDataloader(
        dataset_name='Enformer', dest_path=None, d_output=None,
        tokenizer_name='char', max_length=100000, use_padding=False, return_mask=False, add_eos=False,
        batch_size=16, organism='human'
    )

    loader.setup()
    test_loader = loader.test_dataloader()
    print(len(loader.dataset_train), len(loader.dataset_val), len(loader.dataset_test))

    for sequence, target in test_loader:
        print(sequence.shape, target.shape)
        break

    breakpoint()
