import pretty_midi
import numpy as np
from pathlib import Path
from tqdm import tqdm

import config


# =========================================================
# MIDI EXPORTER
# =========================================================

class MIDIExporter:
    def __init__(self):

        # ===============================
        # GLOBAL CONFIG ALIGNMENT
        # ===============================
        self.bpm = config.BPM
        self.steps_per_bar = config.STEPS_PER_BAR
        self.bars = config.BARS

        # single source of truth
        self.step_duration = config.step_duration_seconds()
        self.total_steps = config.steps_total()

        # drum mapping (GM standard)
        self.map = {
            "kick": 36,
            "snare": 38,
            "hihat": 42
        }

    # =========================================================
    # TRACK BUILDING
    # =========================================================

    def build_track(self, pattern, jitter, pitch):

        notes = []
        n = len(pattern)

        for i in range(n):

            if pattern[i] == 1:

                start = i * self.step_duration + jitter[i]
                start = max(0.0, start)

                # NOTE LENGTH (coherent with grid)
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

    # =========================================================
    # EXPORT SINGLE STIMULUS
    # =========================================================

    def export(self, stim, filename):

        pm = pretty_midi.PrettyMIDI(initial_tempo=self.bpm)

        # time signature fixed
        pm.time_signature_changes.append(
            pretty_midi.TimeSignature(4, 4, 0)
        )

        # =====================================================
        # OPTIONAL SAFETY CHECK (important for research)
        # =====================================================
        expected_steps = config.steps_total()

        for name in ["kick", "snare", "hihat"]:
            if len(stim[name]) < expected_steps:
                raise ValueError(
                    f"{name} pattern too short: "
                    f"{len(stim[name])} < {expected_steps}"
                )

        # =====================================================
        # BUILD INSTRUMENTS
        # =====================================================

        for name in ["kick", "snare", "hihat"]:

            inst = pretty_midi.Instrument(
                program=0,
                is_drum=True
            )

            pattern = stim[name]
            jitter = stim[f"{name}_jitter"]

            inst.notes = self.build_track(
                pattern,
                jitter,
                self.map[name]
            )

            pm.instruments.append(inst)

        # write MIDI
        pm.write(str(filename))


# =========================================================
# BATCH EXPORT
# =========================================================

def export_all(df, stim_cache, out_dir=config.MIDI_DIR):

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