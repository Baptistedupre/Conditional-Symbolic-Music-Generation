# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import note_seq
import numpy as np
from .melody_converter import melody_2bar_converter


def count_measures(note_sequence):
    """Approximate number of measures in the song."""
    splits = note_seq.sequences_lib.split_note_sequence_on_time_changes(note_sequence)
    bars = 0
    for split in splits:
        time_signature = split.time_signatures[0]
        tempo = split.tempos[0]
        quarters_per_bar = 4 * time_signature.numerator / time_signature.denominator
        seconds_per_bar = 60 * quarters_per_bar / tempo.qpm
        num_bars = split.total_time / seconds_per_bar
        bars += num_bars
    return bars


def fix_instruments_for_concatenation(note_sequences):
    """Adjusts instruments for concatenating multitrack measures."""
    instruments = {}
    for i in range(len(note_sequences)):
        for note in note_sequences[i].notes:
            if not note.is_drum:
                if note.program not in instruments:
                    if len(instruments) >= 8:
                        instruments[note.program] = len(instruments) + 2
                    else:
                        instruments[note.program] = len(instruments) + 1
                note.instrument = instruments[note.program]
            else:
                note.instrument = 9


def extract_melodies_raw(note_sequence, keep_longest_split=False):
    """Extracts all melodies in a polyphonic note sequence."""
    splits = note_seq.sequences_lib.split_note_sequence_on_time_changes(note_sequence)
    if keep_longest_split:
        ns = max(splits, key=lambda x: len(x.notes))
        splits = [ns]
    melodies = []
    for split_ns in splits:
        qs = note_seq.sequences_lib.quantize_note_sequence(
            split_ns, steps_per_quarter=4
        )
        instruments = list(set([note.instrument for note in qs.notes]))
        for instrument in instruments:
            melody = note_seq.melodies_lib.Melody()
            try:
                melody.from_quantized_sequence(
                    qs,
                    ignore_polyphonic_notes=True,
                    instrument=instrument,
                    gap_bars=np.inf,
                )
            except note_seq.NonIntegerStepsPerBarError:
                continue
            melody_ns = melody.to_sequence()
            melodies.append(melody_ns)
    return melodies


# TODO: come back to it when the MusicVAE works
def chunks_to_embeddings(sequences, model, data_converter):
    """Convert NoteSequence objects into latent space embeddings.

    Args:
      sequences: A list of NoteSequence objects.
      model: A trained MusicVAE object used for inference.
      data_converter: A data converter used to convert NoteSequence objects into
          tensor encodings for model inference.

    Returns:
      A numpy matrix of shape [len(sequences), latent_dims].
    """
    assert model is not None, "No model provided."

    latent_dims = model._z_input.shape[1]
    idx = []
    non_rest_chunks = []
    zs = np.zeros((len(sequences), latent_dims))
    mus = np.zeros((len(sequences), latent_dims))
    sigmas = np.zeros((len(sequences), latent_dims))
    for i, chunk in enumerate(sequences):
        if len(data_converter.to_tensors(chunk).inputs) > 0:
            idx.append(i)
            non_rest_chunks.append(chunk)
    if non_rest_chunks:
        z, mu, sigma = model.encode(non_rest_chunks)
        assert z.shape == mu.shape == sigma.shape
        for i, mean in enumerate(mu):
            zs[idx[i]] = z[i]
            mus[idx[i]] = mean
            sigmas[idx[i]] = sigma[i]
    return zs, mus, sigmas


# TODO: come back to it when the MusicVAE works
def embeddings_to_chunks(embeddings, model, temperature=1e-3):
    """Decode latent embeddings as NoteSequences.

    Args:
      embeddings: A numpy array of latent embeddings.
      model: A MusicVAE object used for decoding embeddings.

    Returns:
      A list of NoteSequence objects.
    """
    assert model is not None, "No model provided."
    assert len(embeddings) > 0

    reconstructed_chunks = model.decode(
        embeddings, temperature=temperature, length=model._config.hparams.max_seq_len
    )
    assert len(reconstructed_chunks) == len(embeddings)

    embedding_norms = np.linalg.norm(embeddings, axis=1)
    rest_chunk_idx = np.where(embedding_norms == 0)[
        0
    ]  # rests correspond to zero-length embeddings

    for idx in rest_chunk_idx:
        rest_ns = note_seq.NoteSequence()
        rest_ns.total_time = reconstructed_chunks[idx].total_time
        reconstructed_chunks[idx] = rest_ns
    return reconstructed_chunks


class Song(object):
    """Song object used to provide additional abstractions for NoteSequences.

    Attributes:
      note_sequence: A NoteSequence object holding the Song's MIDI data.
      data_converter: A data converter used for preprocessing and tokenization
          for a corresponding MusicVAE model.
      chunk_length: The number of measures in each tokenized chunk of MIDI
          (dependent on the model configuration).
      multitrack: Whether this Song is multitrack or not.
      reconstructed: A boolean describing whether this Song is reconstructed
          from the decoder of a MusicVAE model.
    """

    def __init__(
        self,
        note_sequence,
        data_converter,
        chunk_length=2,
        multitrack=False,
        reconstructed=False,
    ):
        self.note_sequence = note_sequence
        self.data_converter = data_converter
        self.chunk_length = chunk_length
        self.reconstructed = reconstructed
        self.multitrack = multitrack

    def encode(self, model, chunk_length=None, programs=None):
        """Encode song chunks (and full-chunk rests).

        Returns:
          z: (chunks, latent_dims), mu: (chunks, latent_dims), sigma: (chunks, latent_dims).
        """
        chunk_tensors, chunk_sequences = self.chunks(
            chunk_length=chunk_length, programs=programs
        )
        z, means, sigmas = chunks_to_embeddings(
            chunk_sequences, model, self.data_converter
        )
        del chunk_tensors  # unused
        return z

    def chunks(self, chunk_length=None, programs=None, fix_instruments=True):
        """Split and featurize song into chunks of tensors and NoteSequences."""
        assert not self.reconstructed, "Not safe to tokenize reconstructed Songs."

        data = self.note_sequence
        step_size = self.chunk_length
        if chunk_length is not None:
            step_size = chunk_length
        if programs is not None:
            data = self.select_programs(programs)

        # Use the data converter to preprocess sequences
        tensors = self.data_converter.to_tensors(data).inputs[::step_size]
        sequences = self.data_converter.from_tensors(tensors)

        if fix_instruments and self.multitrack:
            fix_instruments_for_concatenation(sequences)

        return tensors, sequences

    def count_chunks(self, chunk_length=None):
        length = self.chunk_length if chunk_length is None else chunk_length
        return count_measures(self.note_sequence) // length

    @property
    def programs(self):
        """MIDI programs used in this song."""
        return list(set([note.program for note in self.note_sequence.notes]))

    def select_programs(self, programs):
        """Keeps selected programs of MIDI (e.g. melody program)."""
        assert len(programs) > 0
        assert all([program >= 0 for program in programs])

        ns = note_seq.NoteSequence()
        ns.CopyFrom(self.note_sequence)
        del ns.notes[:]

        for note in self.note_sequence.notes[:]:
            if note.program in programs:
                new_note = ns.notes.add()
                new_note.CopyFrom(note)
        return ns

    def truncate(self, chunks=0, offset=0):
        """Returns a truncated version of the song.

        Args:
          chunks: The number of chunks in the truncated sequence.
          offset: The offset in chunks to begin truncation.

        Returns:
          A truncated Song object.
        """
        tensors = self.data_converter.to_tensors(self.note_sequence).inputs[
            :: self.chunk_length
        ]
        sequences = self.data_converter.from_tensors(tensors)[offset : offset + chunks]
        fix_instruments_for_concatenation(sequences)
        concat_chunks = note_seq.sequences_lib.concatenate_sequences(sequences)
        return Song(concat_chunks, self.data_converter, chunk_length=self.chunk_length)

    def _count_melody_chunks(self, program):
        """Determines the number of 2-measure chunks using the melody data pipeline."""
        ns = self.select_programs([program])
        tensors = melody_2bar_converter.to_tensors(ns).inputs[::2]
        sequences = melody_2bar_converter.from_tensors(tensors)
        return len(sequences)

    def find_programs(self):
        """Search for the most important MIDI programs in the song."""

        def heuristic(program):
            expected = self.count_chunks(chunk_length=2)
            extracted = self._count_melody_chunks(program)
            if extracted > 0 and abs(extracted - expected) < 0.5 * expected:
                return True
            return False

        midi_programs = self.programs
        top_programs = [p for p in midi_programs if heuristic(p)]
        return top_programs

    def stripped_song(self):
        """A stripped down version using programs found by a special heuristic."""
        top_programs = self.find_programs()
        ns = self.select_programs(top_programs)
        return Song(ns, self.data_converter, self.chunk_length)

    def download(self, filename, preprocessed=True, programs=None):
        """Download song as MIDI file."""
        assert filename is not None, "No filename specified."

        data = self.note_sequence
        if programs is not None:
            data = self.select_programs(programs)

        if (
            not self.reconstructed and preprocessed
        ):  # do not tokenize again if reconstructed
            tensors, chunks = self.chunks(programs=programs)
            del tensors  # unused
            data = note_seq.sequences_lib.concatenate_sequences(chunks)

        note_seq.sequence_proto_to_midi_file(data, filename)

    def play(self, preprocessed=True, programs=None):
        """Play a song with fluidsynth."""
        data = self.note_sequence
        if programs is not None:
            data = self.select_programs(programs)

        if (
            not self.reconstructed and preprocessed
        ):  # do not tokenize again if reconstructed
            tensors, chunks = self.chunks(programs=programs)
            del tensors  # unused
            data = note_seq.sequences_lib.concatenate_sequences(chunks)

        note_seq.play_sequence(data, synth=note_seq.fluidsynth)
        return data
