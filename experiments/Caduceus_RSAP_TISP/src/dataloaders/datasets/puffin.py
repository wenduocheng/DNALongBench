from itertools import islice
from functools import partial
import os
import sys
import functools
# import json
# from pathlib import Path
# from pyfaidx import Fasta
# import polars as pl
# import pandas as pd
import torch
from random import randrange, random
import numpy as np
from pathlib import Path

from src.dataloaders.base import SequenceDataset, default_data_path
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer

sys.path.append('dependencies/selene')
import pyBigWig
import tabix
from selene_sdk.sequences import Genome
from selene_sdk.targets import Target
from selene_sdk.samplers import RandomPositionsSampler
from selene_sdk.samplers.dataloader import SamplerDataLoader

"""

Puffin Dataset

"""

char_list = ['A', 'C', 'G', 'T', 'N']
idx2char = {i: char_list[i] for i in range(4)}
char2idx = {char_list[i]: i for i in range(4)}


def convert_from_one_hot(sequence):
    x_str = "" 
    for x in sequence:
        if x.max() == 0.25:
            x_str += 'N'
        else:
            x_str += idx2char[x.argmax()]
    return x_str


class GenomicSignalFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """
    def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None, 
        replacement_indices=None, replacement_scaling_factors=None):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors

            
        self.n_features = len(features)
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist)  for blacklist in self.blacklists]
            self.initialized=True
            
        wigmat = np.vstack([c.values(chrom, start, end, numpy=True)
                           for c in self.data])
        
        if self.blacklists is not None:
            if self.replacement_indices is None:
                for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s)-start,0): int(e)-start] = 0
            else:
                for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(self.blacklists, self.blacklists_indices, self.replacement_indices, self.replacement_scaling_factors):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[blacklist_indices, np.fmax(int(s)-start,0): int(e)-start] = wigmat[replacement_indices, np.fmax(int(s)-start,0): int(e)-start] * replacement_scaling_factor

        if nan_as_zero:
            wigmat[np.isnan(wigmat)]=0
        return wigmat



class PuffinDataset(torch.utils.data.IterableDataset):

    '''
    Puffin Dataset
    '''

    def __init__(
        self,
        split,
        max_length,
        dataset_name="Puffin",
        d_output=1, # default regression
        track=None,
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        return_mask=False,
    ):

        self.dataset_name = dataset_name
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.return_mask = return_mask
        self.split = split
        self.dest_path = str(dest_path)
        self.track = track

        if isinstance(dest_path, Path):
            dest_path = str(dest_path) + '/'

        # use Path object
        self.input_files = [dest_path + "agg.plus.bw.bedgraph.bw",
                dest_path + "agg.encodecage.plus.v2.bedgraph.bw",
                dest_path + "agg.encoderampage.plus.v2.bedgraph.bw",
                dest_path + "agg.plus.grocap.bedgraph.sorted.merged.bw",
                dest_path + "agg.plus.allprocap.bedgraph.sorted.merged.bw",
                dest_path + "agg.minus.allprocap.bedgraph.sorted.merged.bw",
                dest_path + "agg.minus.grocap.bedgraph.sorted.merged.bw",
                dest_path + "agg.encoderampage.minus.v2.bedgraph.bw",
                dest_path + "agg.encodecage.minus.v2.bedgraph.bw",
                dest_path + "agg.minus.bw.bedgraph.bw"]

        tfeature = GenomicSignalFeatures(self.input_files,
                ['cage_plus','encodecage_plus','encoderampage_plus', 'grocap_plus','procap_plus','procap_minus','grocap_minus'
                ,'encoderampage_minus', 'encodecage_minus',
                'cage_minus'],
                (100000,),
                [dest_path + "fantom.blacklist8.plus.bed.gz",dest_path + "fantom.blacklist8.minus.bed.gz"],
                [0,9], [1,8], [0.61357, 0.61357])
        
        self.genome = Genome(
                    input_path=dest_path + "Homo_sapiens.GRCh38.dna.primary_assembly.fa",
                    blacklist_regions= 'hg38'
                )

        self.noblacklist_genome = Genome(
                    input_path=dest_path + "Homo_sapiens.GRCh38.dna.primary_assembly.fa" )

        self.sampler = RandomPositionsSampler(
                            reference_sequence=self.genome,
                            target= tfeature,
                            features = [''],
                            test_holdout=['chr8', 'chr9'],
                            validation_holdout= ['chr10'],
                            sequence_length= 100_000,
                            center_bin_to_predict= 100_000,
                            position_resolution=1,
                            random_shift=0,
                            random_strand=False
                            )

        self.sampler.mode = "validate" if self.split == "val" else self.split

        seed = 3
        np.random.seed(seed)

        self.buffer = []
        self.buffer_size = 1024

    def get_test_data(self):
        slide_window = 50_000
        sequence_length = self.max_length
        ignore_last_len = 25_000
        remove_unk_interval = 1000


        def check_unk(position):
            start_position = position - remove_unk_interval
            end_position = position + remove_unk_interval
            sub_seq = self.sampler.reference_sequence.get_sequence_from_coords(chrom, start_position, end_position)
            if 'N' in sub_seq:
                return True
            return False


        test_position_index = []
        for index, (chrom, len_chrom) in enumerate(self.sampler.reference_sequence.get_chr_lens()):
            if chrom == 'chr8' or chrom == 'chr9':

                length = len_chrom - ignore_last_len
                for position in range(sequence_length, length - sequence_length, slide_window):
                    start = position - sequence_length // 2
                    end = position + sequence_length // 2
                    left = check_unk(start)
                    right = check_unk(end)
                    if left == True or right == True:
                        continue
                    sequence = self.sampler.reference_sequence.get_sequence_from_coords(chrom, start, end)
                    if 'N' in sequence:
                        continue    
                    test_position_index.append((chrom, position))
        print("Test Dataset", len(test_position_index))
        return test_position_index

    def update_buffer(self):
        # print("Update Buffer")
        sequence, y = self.sampler.sample(batch_size=self.buffer_size)   # (1, 100000, 4), (1, 10, 100000)
        x_str = [convert_from_one_hot(s) for s in sequence]

        seq = self.tokenizer(x_str,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids
        seq_ids = torch.LongTensor(seq_ids)
        mask = torch.BoolTensor(seq['attention_mask'])

        # need to wrap in list
        target = torch.FloatTensor(y)
        target = target.permute(0, 2, 1)  # (bz, 10, 100000) -> (bz, 100000, 10)
        if self.track is not None:
            target = target[:, :, self.track].unsqueeze(-1)

        self.buffer = [(seq_ids[i], target[i], mask[i]) for i in range(self.buffer_size)]


    def generator(self):
        while True:
            self.update_buffer()
            
            for i in range(self.buffer_size):
                seq_ids, target, mask = self.buffer[i]

                if self.return_mask:
                    yield seq_ids, target, {'mask': mask}
                else:
                    yield seq_ids, target

        
    def valid_generator(self):
        test_position_index = self.get_test_data()
        # test_dataset = [self.sampler._retrieve(chrom, position) for chrom, position in test_position_index]
        # print("Finish Test Dataset Generation")

        for chrom, position in test_position_index:
            return_result = self.sampler._retrieve(chrom, position)
            if return_result is None:
                continue
            sequence, y = return_result
            x_str = convert_from_one_hot(sequence)

            seq = self.tokenizer(x_str,
                add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
                padding="max_length" if self.use_padding else "do_not_pad",
                max_length=self.max_length,
                truncation=True,
            )
            seq_ids = seq["input_ids"]
            seq_ids = torch.LongTensor(seq_ids)
            target = torch.FloatTensor(y)
            target = target.permute(1,0)  # (10, 100000) -> (100000, 10)

            if self.track is not None:
                target = target[:, self.track].unsqueeze(-1)

            if self.return_mask:
                yield seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
            else:
                yield seq_ids, target

    def test_generator(self):
        test_file = os.path.join(self.dest_path, "evaluation/allpred.str.npy")
        test_data = np.load(test_file, allow_pickle=True)
        print("Finish load test dataset")
        for x_str in test_data:
            seq = self.tokenizer(x_str,
                add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
                padding="max_length" if self.use_padding else "do_not_pad",
                max_length=self.max_length,
                truncation=True,
            )
            seq_ids = seq["input_ids"]
            seq_ids = torch.LongTensor(seq_ids)
            target = torch.zeros(self.max_length, 10)

            if self.track is not None:
                target = target[:, self.track].unsqueeze(-1)

            if self.return_mask:
                yield seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
            else:
                yield seq_ids, target


        
    def __iter__(self):
        if self.split == "train":
            return self.generator()
        elif self.split == "val":
            return self.valid_generator()
        else:
            return self.test_generator()








if __name__ == '__main__':
    """Quick test loading dataset.
    
    example
    python -m src.dataloaders.datasets.puffin
    
    """

    max_length = 100_000  # max len of seq grabbed
    use_padding = False
    dest_path = default_data_path / "puffin"
    return_mask = False
    add_eos = False
    padding_side = 'right'   

    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=max_length,
        add_special_tokens=False,
        padding_side=padding_side,
    )

    ds = PuffinDataset(
        max_length = max_length,
        use_padding = use_padding,
        split = 'test', # 
        track = 1,
        tokenizer=tokenizer,
        tokenizer_name='char',
        dest_path=dest_path,
        return_mask=return_mask,
        add_eos=add_eos,
    )

    # it = iter(ds)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=25, num_workers=0)
    for sequence, target in train_loader:
        print(sequence.shape,target.shape)
        # break
    # elem = next(it)
    # print('elem[0].shape', elem[0].shape)
    # print(elem)
    breakpoint()