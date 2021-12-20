class Agent():
    def __init__(self):
        return

    def pre_episode_setup(self):
        raise NotImplementedError("Implement it in the inheriting class.")

    def get_action(self, state):
        raise NotImplementedError("Implement it in the inheriting class.")

    def get_result(self, result: int):
        raise NotImplementedError("Implement it in the inheriting class.")
