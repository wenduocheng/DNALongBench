import torch
import torch.nn.functional as F
import numpy as np

import os
import json
import pandas as pd
import pyfaidx
import kipoiseq
import functools
from kipoiseq import Interval
import tensorflow as tf

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


# Data paths
human_fasta_path = '../data/Enformer/hg38.ml.fa'
mouse_fasta_path = '../data/Enformer/mm38.ml.fa'
data_path = '../data/Enformer/data'


SEQUENCE_LENGTH = 196_608
BIN_SIZE = 128
TARGET_LENGTH = 896


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


class BasenjiDataSet(torch.utils.data.IterableDataset):
    @staticmethod
    def get_organism_path(organism):
        return os.path.join(data_path, organism)
    @classmethod
    def get_metadata(cls, organism):
        # Keys:
        # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
        # pool_width, crop_bp, target_length
        path = os.path.join(cls.get_organism_path(organism), 'statistics.json')
        with tf.io.gfile.GFile(path, 'r') as f:
            return json.load(f)
    @staticmethod
    def one_hot_encode(sequence):
        return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

    @classmethod
    def get_tfrecord_files(cls, organism, subset):
        # Sort the values by int(*).
        return sorted(tf.io.gfile.glob(os.path.join(
                cls.get_organism_path(organism), 'tfrecords', f'{subset}-*.tfr'
            )), key=lambda x: int(x.split('-')[-1].split('.')[0]))

    @property
    def num_channels(self):
        metadata = self.get_metadata(self.organism)
        return metadata['num_targets']

    @staticmethod
    def deserialize(serialized_example, metadata):
        """Deserialize bytes stored in TFRecordFile."""
        # Deserialization
        feature_map = {
                'sequence': tf.io.FixedLenFeature([], tf.string),    # Ignore this, resize our own bigger one
                'target': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_example(serialized_example, feature_map)
        sequence = tf.io.decode_raw(example['sequence'], tf.bool)
        sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
        sequence = tf.cast(sequence, tf.float32)

        target = tf.io.decode_raw(example['target'], tf.float16)
        target = tf.reshape(target,(metadata['target_length'], metadata['num_targets']))
        target = tf.cast(target, tf.float32)

        return {'sequence_old': sequence,
                        'target': target}

    @classmethod
    def get_dataset(cls, organism, subset, num_threads=8):
        metadata = cls.get_metadata(organism)
        dataset = tf.data.TFRecordDataset(cls.get_tfrecord_files(organism, subset),
                                                                            compression_type='ZLIB',
                                                                            num_parallel_reads=num_threads).map(
                                                                                    functools.partial(cls.deserialize, metadata=metadata)
                                                                            )
        return dataset

    def __init__(self, organism:str, subset:str, seq_len:int, fasta_path:str, n_to_test:int = -1):
        assert subset in {"train", "valid", "test"}
        assert organism in {"human", "mouse"}
        self.organism = organism
        self.subset = subset
        self.base_dir = self.get_organism_path(organism)
        self.seq_len = seq_len
        self.fasta_reader = FastaStringExtractor(fasta_path)
        self.n_to_test = n_to_test
        with tf.io.gfile.GFile(f"{self.base_dir}/sequences.bed", 'r') as f:
            region_df = pd.read_csv(f, sep="\t", header=None)
            region_df.columns = ['chrom', 'start', 'end', 'subset']
            self.region_df = region_df.query('subset==@subset').reset_index(drop=True)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is None, "Only support single process loading"
        # If num_threads > 1, the following will actually shuffle the inputs! luckily we catch this with the sequence comparison
        basenji_iterator = self.get_dataset(self.organism, self.subset, num_threads=1).as_numpy_iterator()
        for i, records in enumerate(basenji_iterator):
            loc_row = self.region_df.iloc[i]
            target_interval = Interval(loc_row['chrom'], loc_row['start'], loc_row['end'])
            sequence_one_hot = self.one_hot_encode(self.fasta_reader.extract(target_interval.resize(self.seq_len)))
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
            yield {
                    "sequence": sequence_one_hot,
                    "target": records["target"],
            }



organism="human"

def get_test_set(organism):
    max_steps=-1
    fasta_path = human_fasta_path if organism == "human" else mouse_fasta_path
    ds = BasenjiDataSet(organism, "test", SEQUENCE_LENGTH, fasta_path)

    test_human_loader = torch.utils.data.DataLoader(ds, num_workers=0, batch_size=32, drop_last=False)

    alltruth = []
    for i, batch in enumerate(test_human_loader):
        alltruth.append(batch["target"])

    alltarget = np.concatenate([x.numpy() for x in alltruth])

    np.save(f"../data/Enformer/{organism}_test.npy", alltarget)
    print(f"Saved {organism} test set to ../data/Enformer/{organism}_test.npy")


if __name__ == "__main__":
    get_test_set("human")
    get_test_set("mouse")