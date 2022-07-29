"""ReplayRecord
"""


class ReplayRecord:
    """ReplayRecord class"""

    def __init__(
        self,
        path,
        filename,
        dataset,
        genuine,
        client,
        session,
        attack,
        support,
        device,
        presentation,
        light,
        step,
        light_condition,
        landmarks,
    ):
        self.path = path
        self.filename = filename
        self.dataset = dataset
        self.genuine = genuine
        self.client = client
        self.session = session
        self.attack = attack
        self.support = support
        self.device = device
        self.presentation = presentation
        self.light = light
        self.step = step
        self.light_condition = light_condition
        self.landmarks = landmarks

    def __repr__(self):
        return f"[{self.path}-{self.genuine}"
