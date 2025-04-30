from itertools import islice
from functools import partial
import os
import functools
# import json
# from pathlib import Path
# from pyfaidx import Fasta
# import polars as pl
# import pandas as pd
import torch
from random import randrange, random
import numpy as np
import pandas as pd
from pathlib import Path

import json
import pyfaidx
import kipoiseq
from kipoiseq import Interval
import functools

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.dataloaders.base import default_data_path

"""

Genomic Benchmarks Dataset, from:
https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks


"""


# helper functions


SEQUENCE_LENGTH = 196_608
BIN_SIZE = 128
TARGET_LENGTH = 896

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

def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

def deserialize(serialized_example, metadata):
    """Deserialize bytes stored in TFRecordFile."""
    # Deserialization
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),  # Ignore this, resize our own bigger one
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)

    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (metadata['target_length'], metadata['num_targets']))
    target = tf.cast(target, tf.float32)

    return {'sequence_old': sequence,
            'target': target}

class FastaStringExtractor:
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


class EnformerDataset(torch.utils.data.IterableDataset):

    '''
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    Returns a generator that retrieves the sequence.
    '''

    def __init__(
        self,
        split,
        max_length,
        dataset_name="Enformer",
        organism='human',
        d_output=1, 
        l_output=1, 
        seq_len=SEQUENCE_LENGTH,
        n_to_test=-1,
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
        self.l_output = l_output  # needed for decoder to grab
        self.return_mask = return_mask
        self.split = "test" if split == "val" else split
        self.organism = organism
        self.seq_len = seq_len
        self.n_to_test = n_to_test

        # use Path object
        base_path = Path(dest_path) / "data"
        self.base_dir = str(base_path / organism)

        
        if organism == 'human':
            fasta_path = dest_path / 'hg38.ml.fa'
        elif organism == 'mouse':
            fasta_path = dest_path / 'mm10.ml.fa'
        self.fasta_reader = FastaStringExtractor(fasta_path)
        with tf.io.gfile.GFile(f"{self.base_dir}/sequences.bed", 'r') as f:
            region_df = pd.read_csv(f, sep="\t", header=None)
            region_df.columns = ['chrom', 'start', 'end', 'subset']
            self.region_df = region_df.query('subset==@self.split').reset_index(drop=True)
        
        self.metadata = self.get_metadata()
        self.num_targets = self.metadata['num_targets']

    def __len__(self):
        return len(self.region_df)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is None, "Only support single process loading"
        # If num_threads > 1, the following will actually shuffle the inputs! luckily we catch this with the sequence comparison
        basenji_iterator = self.get_dataset(num_threads=16).as_numpy_iterator()
        for i, records in enumerate(basenji_iterator):
            loc_row = self.region_df.iloc[i]
            target_interval = Interval(loc_row['chrom'], loc_row['start'], loc_row['end'])
            sequence = self.fasta_reader.extract(target_interval.resize(self.seq_len))
            sequence_one_hot = one_hot_encode(sequence)
            if self.n_to_test >= 0 and i < self.n_to_test:
                old_sequence_onehot = records["sequence_old"]
                if old_sequence_onehot.shape[0] > sequence_one_hot.shape[0]:
                    diff = old_sequence_onehot.shape[0] - sequence_one_hot.shape[0]
                    trim = diff//2
                    np.testing.assert_equal(old_sequence_onehot[trim:(-trim)], sequence_one_hot)
                elif sequence_one_hot.shape[0] > old_sequence_onehot.shape[0]:
                    diff = sequence_one_hot.shape[0] - old_sequence_onehot.shape[0]
                    trim = diff//2
                    np.testing.assert_equal(old_sequence_onehot, sequence_one_hot[trim:(-trim)])
                else:
                    np.testing.assert_equal(old_sequence_onehot, sequence_one_hot)

            yield self.convert(sequence_one_hot, records["target"])

    def convert(self, x, y):
        x_str = convert_from_one_hot(x)

        seq = self.tokenizer(x_str,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids 
        seq_ids = torch.LongTensor(seq_ids) # [SEQUENCE_LENGTH]

        # need to wrap in list
        target = torch.FloatTensor(y)   # [TARGET_LENGTH, 5313]

        # print(x.shape, y.shape)

        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target
    
    def get_metadata(self):
        # Keys:
        # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
        # pool_width, crop_bp, target_length
        path = os.path.join(self.base_dir, 'statistics.json')
        with tf.io.gfile.GFile(path, 'r') as f:
            return json.load(f)

    def get_tfrecord_files(self):
        # Sort the values by int(*).
        return sorted(tf.io.gfile.glob(os.path.join(
                    self.base_dir, 'tfrecords', f'{self.split}-*.tfr'
                )), key=lambda x: int(x.split('-')[-1].split('.')[0]))

    def get_dataset(self, num_threads=8):
        dataset = tf.data.TFRecordDataset(self.get_tfrecord_files(),
                                        compression_type='ZLIB',
                                        num_parallel_reads=num_threads).map(
                                            functools.partial(deserialize, metadata=self.metadata)
                                        )
        return dataset



if __name__ == '__main__':
    """Quick test loading dataset.
    
    example
    python -m src.dataloaders.datasets.enformer
    
    """

    max_length = SEQUENCE_LENGTH  # max len of seq grabbed
    use_padding = False
    dest_path = default_data_path / "Enformer"
    return_mask = False
    add_eos = False
    padding_side = 'right'   
    organism = 'human'


    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=max_length,
        add_special_tokens=False,
        padding_side=padding_side,
    )

    ds = EnformerDataset(
        max_length = max_length,
        use_padding = use_padding,
        split = 'train', # 
        tokenizer=tokenizer,
        tokenizer_name='char',
        dest_path=dest_path,
        return_mask=return_mask,
        add_eos=add_eos,
    )

    # it = iter(ds)
    device = "cpu"
    train_loader = torch.utils.data.DataLoader(ds, batch_size=1000, num_workers=0)
    for i, (sequence, target) in enumerate(train_loader):
        sequence = sequence.to(device)
        target = target.to(device)
        # print(sequence.shape,target.shape)
        print(i)
        # break
    # elem = next(it)
    # print('elem[0].shape', elem[0].shape)
    # print(elem)
    # breakpoint()