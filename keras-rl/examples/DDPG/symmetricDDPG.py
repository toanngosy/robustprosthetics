from rl.agents import DDPGAgent

class SymmetricDDPGAgent(DDPGAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def select_action(self, state):
        print("ahihi")
        return 0
