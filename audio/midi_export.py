"""
audio/midi_export.py
====================
Export MIDI des stimuli générés.

Vélocités : les vélocités sont désormais portées par le stim lui-même
(kick_vel, snare_vel, hihat_vel, bass_vel) et humanisées par MicroTiming
selon E_mv. Fallback sur les constantes DEFAULT_VELOCITY si absent (rétro-compat).

Pad harmonique :
    Un accord Am7 tenu est ajouté comme couche invariante sur tous les stimuli
    (config.PAD_ENABLED). Étant strictement constant entre conditions, il ne
    confond aucune variable manipulée — même logique que la basse (cf. §2.2.2).
    Référence : Witek et al. (2014) utilisent un fond harmonique stable.
"""

import pretty_midi
import numpy as np
from pathlib import Path
from tqdm import tqdm

import config


# =========================================================
# VÉLOCITÉS PAR DÉFAUT (fallback si pas de *_vel dans le stim)
# =========================================================

DEFAULT_VELOCITY = {
    "kick":  95,
    "bass":  85,
    "snare": 90,
    "hihat": 80,
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

    # ── Voix percussives ──────────────────────────────────

    def build_track(
        self,
        pattern:     np.ndarray,
        jitter:      np.ndarray,
        pitch:       int,
        vel:         np.ndarray | None = None,
        default_vel: int = 90,
    ) -> list[pretty_midi.Note]:
        """
        Construit une liste de notes MIDI pour une voix percussive (pitch fixe).

        Args:
            pattern     : pattern binaire (0/1)
            jitter      : décalages temporels en secondes
            pitch       : numéro MIDI fixe pour tous les hits
            vel         : vecteur de vélocités (0–127) — humanisé si fourni
            default_vel : vélocité fixe si vel est None (rétro-compat)
        """
        if len(pattern) != len(jitter):
            raise ValueError("Pattern et jitter doivent avoir la même longueur")

        notes = []
        for i, hit in enumerate(pattern):
            if hit != 1.0:
                continue
            start    = max(0.0, i * self.step_duration + jitter[i])
            end      = start + self.step_duration * 0.9
            velocity = int(np.clip(vel[i], 1, 127)) if vel is not None else default_vel
            notes.append(pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start,
                end=end,
            ))
        return notes

    def build_bass_track(
        self,
        pattern:     np.ndarray,
        jitter:      np.ndarray,
        pitch_array: np.ndarray,
        vel_array:   np.ndarray,
        dur_array:   np.ndarray,
    ) -> list[pretty_midi.Note]:
        """Basse avec pitch, vélocité et durée variables par step."""
        notes = []
        for i, hit in enumerate(pattern):
            if hit <= 0.0:
                continue
            start    = max(0.0, i * self.step_duration + jitter[i])
            duration = max(self.step_duration * 0.1, dur_array[i])
            end      = start + duration
            velocity = int(np.clip(vel_array[i], 1, 127))
            notes.append(pretty_midi.Note(
                velocity=velocity,
                pitch=int(pitch_array[i]),
                start=start,
                end=end,
            ))
        return notes

    # ── Pad harmonique invariant ──────────────────────────

    def build_pad_track(
        self,
        total_duration: float,
    ) -> pretty_midi.Instrument:
        """
        Construit le pad harmonique Am7 invariant.

        Accord Am7 = A2 (45), E3 (52), G3 (55), B3 (59)
        Tenu sur toute la durée du stimulus, vélocité faible (config.PAD_VELOCITY).

        Le pad est strictement constant entre tous les stimuli :
        il ne porte aucune information différentielle entre conditions
        et ne peut donc pas confondre les variables manipulées.
        """
        inst = pretty_midi.Instrument(
            program=config.PAD_PROGRAM,
            is_drum=False,
            name="Pad Am7",
        )

        start = 0.0
        end   = total_duration

        for pitch in config.PAD_PITCHES:
            inst.notes.append(pretty_midi.Note(
                velocity=config.PAD_VELOCITY,
                pitch=pitch,
                start=start,
                end=end,
            ))

        return inst

    # ── Export principal ──────────────────────────────────

    def export(self, stim: dict, filename) -> None:
        pm = pretty_midi.PrettyMIDI(initial_tempo=self.bpm)
        pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0))

        expected_steps = config.total_steps()
        for name in ("kick", "bass", "snare", "hihat"):
            if len(stim[name]) != expected_steps:
                raise ValueError(
                    f"{name} length mismatch: "
                    f"{len(stim[name])} != {expected_steps}"
                )

        # ── Percussions ───────────────────────────────────
        for name in ("kick", "snare", "hihat"):
            inst      = pretty_midi.Instrument(program=0, is_drum=True)
            # Préférence aux vélocités humanisées du stim (v2),
            # fallback sur DEFAULT_VELOCITY (rétro-compat v1)
            vel_array = stim.get(f"{name}_vel", None)

            inst.notes = self.build_track(
                stim[name],
                stim[f"{name}_jitter"],
                self.map[name],
                vel=vel_array,
                default_vel=DEFAULT_VELOCITY[name],
            )
            pm.instruments.append(inst)

        # ── Basse ─────────────────────────────────────────
        bass_inst = pretty_midi.Instrument(program=33, is_drum=False, name="Bass")
        bass_inst.notes = self.build_bass_track(
            stim["bass"],
            stim["bass_jitter"],
            pitch_array=stim["bass_pitch"],
            vel_array=stim["bass_vel"],
            dur_array=stim["bass_dur"],
        )
        pm.instruments.append(bass_inst)

        # ── Pad harmonique invariant ──────────────────────
        if config.PAD_ENABLED:
            total_dur = config.stimulus_duration_seconds()
            pad_inst  = self.build_pad_track(total_dur)
            pm.instruments.append(pad_inst)

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