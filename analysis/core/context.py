class AnalysisContext:

    def __init__(self, run_dir, dataset, seed=42, config=None):

        self.run_dir = run_dir
        self.dataset = dataset
        self.seed = seed

        # 🔥 IMPORTANT: unified config access
        self.config = config or {}

        self.cache = {}

        from analysis.io.run_manager import RunManager
        self.run_manager = RunManager(run_dir)

    def log(self, msg: str):
        from datetime import datetime
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [ANALYSIS] {msg}")