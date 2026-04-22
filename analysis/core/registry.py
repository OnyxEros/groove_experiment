STEP_REGISTRY = {}


def register_step(name):
    def wrapper(cls):
        STEP_REGISTRY[name] = cls
        return cls
    return wrapper


def get_step(name):
    if name not in STEP_REGISTRY:
        raise ValueError(
            f"Unknown step '{name}'. Available: {list(STEP_REGISTRY.keys())}"
        )
    return STEP_REGISTRY[name]


import pkgutil
import importlib
import analysis.steps
import traceback


def load_steps():

    print("[REGISTRY] loading steps...")

    for _, name, _ in pkgutil.iter_modules(analysis.steps.__path__):

        try:
            print(f"[REGISTRY] importing {name}")
            importlib.import_module(f"analysis.steps.{name}")

        except Exception as e:
            print(f"[REGISTRY] FAILED {name}: {e}")
            traceback.print_exc()

    print(f"[REGISTRY] loaded: {list(STEP_REGISTRY.keys())}")