from enum import Enum

class TruthyFalseyClassifier:
    def __init__(self, value):
        self.value = value

    def get_classification(self):
        return (Classification.TRUE if bool(self.value)
                                    else Classification.FALSE)

class Classification(Enum):
    TRUE  = 1
    FALSE = 0