import pretty_midi
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import MIDI_DIR


class MIDIExporter:
    def __init__(self, bpm=120, subdivision=4):

        self.bpm = bpm
        self.subdivision = subdivision
        self.step_duration = 60 / (bpm * subdivision)

        self.map = {
            "kick": 36,
            "snare": 38,
            "hihat": 42
        }

    def build_track(self, pattern, jitter, pitch):

        notes = []
        n = len(pattern)

        for i in range(n):
            if pattern[i] == 1:

                start = i * self.step_duration + jitter[i]
                start = max(0, start)

                end = start + self.step_duration * 0.9

                notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=start,
                        end=end
                    )
                )

        return notes

    def export(self, stim, filename):

        repeat_bars = 8
        steps_per_bar = 16

        pattern_length = len(stim["kick"])

        if pattern_length < steps_per_bar:
            raise ValueError("Pattern trop court (moins d'1 mesure)")

        def repeat(x):
            return np.tile(x, repeat_bars)

        pm = pretty_midi.PrettyMIDI(initial_tempo=self.bpm)

        pm.time_signature_changes.append(
            pretty_midi.TimeSignature(4, 4, 0)
        )

        for name in ["kick", "snare", "hihat"]:

            inst = pretty_midi.Instrument(
                program=0,
                is_drum=True
            )

            pattern = repeat(stim[name])
            jitter = repeat(stim[f"{name}_jitter"])

            inst.notes = self.build_track(
                pattern,
                jitter,
                self.map[name]
            )

            pm.instruments.append(inst)

        pm.write(str(filename))


def export_all(df, stim_cache, out_dir=MIDI_DIR):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exporter = MIDIExporter()

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="🎼 Exporting MIDI",
        unit="file"
    ):

        stim = stim_cache[row["id"]]

        filename = out_dir / f"stim_{int(row['id']):04d}.mid"

        exporter.export(stim, filename)


