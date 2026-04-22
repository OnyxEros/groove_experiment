from analysis.core.registry import get_step


def build_pipeline(step_names):

    return [
        get_step(name)()
        for name in step_names
    ]