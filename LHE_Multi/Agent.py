class Agent():
    def __init__(self):
        return

    def get_action(self, state):
        raise NotImplementedError("Implement it in the inheriting class.")

    def get_result(self, result: int):
        raise NotImplementedError("Implement it in the inheriting class.")
