"""RoseRecord
"""


class RoseRecord:
    """RoseRecord class"""

    def __init__(self, path, filename, l, s, d, x, e, p, n, landmarks):
        self.path = path
        self.filename = filename
        self.l = l  # pylint: disable=invalid-name
        self.s = s  # pylint: disable=invalid-name
        self.d = d  # pylint: disable=invalid-name
        self.x = x  # pylint: disable=invalid-name
        self.e = e  # pylint: disable=invalid-name
        self.p = p  # pylint: disable=invalid-name
        self.n = n  # pylint: disable=invalid-name
        self.landmarks = landmarks

    def __repr__(self):
        return f"[{self.path}, {self.filename}, {self.l}, {self.s}, {self.d}, {self.x}, {self.e}, {self.p}, {self.n}, {self.landmarks.shape()}]"
