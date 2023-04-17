"""ReplayRecord
"""


class ReplayAttackRecord:
    """ReplayRecord class"""

    def __init__(self, path, filename, dataset, genuine, client):
        self.path = path
        self.filename = filename
        self.dataset = dataset
        self.genuine = genuine
        self.client = client

    def __repr__(self):
        return f"[{self.path}"
