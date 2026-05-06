"""
audio/midi_export.py
====================
Export MIDI des stimuli générés.

Vélocités : si le stimulus contient des vecteurs *_vel (v3+), ils sont utilisés.
Sinon fallback sur une vélocité fixe par voix (rétro-compatible v1/v2).
Les vélocités fixes par défaut reflètent la hiérarchie perceptive :
kick (95) > snare (90) > hihat (75).
"""

import pretty_midi
import numpy as np
from pathlib import Path
from tqdm import tqdm

import config


# =========================================================
# VELOCITÉS PAR DÉFAUT (fallback si pas de *_vel dans le stim)
# =========================================================

DEFAULT_VELOCITY = {
    "kick":  95,
    "bass":  85,
    "snare": 90,
    "hihat": 75,
}


# =========================================================
# EXPORTER
# =========================================================

class MIDIExporter:

    def __init__(self):
        self.bpm           = config.BPM
        self.step_duration = config.step_duration_seconds()
        self.map = {
            "kick":  36,
            "bass":  33,
            "snare": 38,
            "hihat": 42,
        }

    def build_track(
        self,
        pattern:      np.ndarray,
        jitter:       np.ndarray,
        pitch:        int = None,
        pitch_array:  np.ndarray = None,
        vel:          np.ndarray | None = None,
        default_vel:  int = 90,
    ) -> list[pretty_midi.Note]:
        """
        Args:
            pattern     : pattern binaire (0/1)
            jitter      : décalages temporels en secondes
            pitch       : numéro MIDI de la note
            vel         : vecteur de vélocités optionnel (0–127)
            default_vel : vélocité fixe si vel est None
        """
        if len(pattern) != len(jitter):
            raise ValueError("Pattern et jitter doivent avoir la même longueur")

        notes = []
        for i, hit in enumerate(pattern):
            if hit != 1.0:
                continue

            start     = max(0.0, i * self.step_duration + jitter[i])
            end       = start + self.step_duration * 0.9
            velocity  = int(np.clip(vel[i], 1, 127)) if vel is not None else default_vel
            pitch_val = int(pitch_array[i]) if pitch_array is not None else pitch
            notes.append(pretty_midi.Note(
                velocity=velocity,
                pitch=pitch_val,
                start=start,
                end=end,
            ))

        return notes

    def export(self, stim: dict, filename) -> None:
        pm = pretty_midi.PrettyMIDI(initial_tempo=self.bpm)
        pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0))

        expected_steps = config.total_steps()
        for name in ("kick", "snare", "hihat"):
            if len(stim[name]) != expected_steps:
                raise ValueError(
                    f"{name} length mismatch: "
                    f"{len(stim[name])} != {expected_steps}"
                )

        for name in ("kick", "snare", "hihat"):
            inst      = pretty_midi.Instrument(program=0, is_drum=True)
            vel_array = stim.get(f"{name}_vel", None)   # None si absent (v1/v2)

            inst.notes = self.build_track(
                stim[name],
                stim[f"{name}_jitter"],
                self.map[name],
                vel=vel_array,
                default_vel=DEFAULT_VELOCITY[name],
            )
            pm.instruments.append(inst)

        # ── Basse (canal instrument, program 33 = Finger Bass) ─
        bass_inst = pretty_midi.Instrument(program=33, is_drum=False)
        bass_inst.notes = self.build_track(
            stim["bass"],
            stim["bass_jitter"],
            pitch_array=stim["bass_pitch"],
            vel=None,
            default_vel=config.BASS_VELOCITY,
        )
        pm.instruments.append(bass_inst)

        pm.write(str(filename))


# =========================================================
# BATCH EXPORT
# =========================================================

def export_all(df, stim_cache, out_dir=config.MIDI_DIR):
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exporter = MIDIExporter()

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="🎼 Exporting MIDI",
        unit="file",
    ):
        stim     = stim_cache[row["id"]]
        filename = out_dir / f"stim_{int(row['id']):04d}.mid"
        exporter.export(stim, filename)