from analysis.pipelines.audio_space import run_audio_pipeline
from analysis.pipelines.groove_space import run_groove_pipeline


def run_full_analysis(mp3_dir, save=True):

    df, groove_emb, groove_reducer = run_groove_pipeline(save)

    audio_emb, paths, audio_reducer = run_audio_pipeline(mp3_dir, save)

    return {
        "groove": groove_emb,
        "audio": audio_emb,
        "paths": paths
    }