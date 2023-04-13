"""ReplayRecord
"""


class ReplayAttackRecord:
    """ReplayRecord class"""

    def __init__(self, path, filename, dataset, genuine):
        self.path = path
        self.filename = filename
        self.dataset = dataset
        self.genuine = genuine

    def __repr__(self):
        return f"[{self.path}"
