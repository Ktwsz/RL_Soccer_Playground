from env_base import EnvironmentBase

class Environment(EnvironmentBase):
    def action(self, action):
        norm = self.get_norm()
        super().action(action)
        
        return norm
    
    def get_state(self):
        state = []
        for i in range(self.width*self.height):
            for j in range(8):
                state.append((self.visited[i//self.height][i%self.height]//(2**j))%2)

        for i in range(self.width*self.height):
            state.append(0 if self.pos[0] != i//self.height and self.pos[1] != i%self.height else 1)
        state.append(self.player)
        return state
    
    def get_norm(self):
        actions = [1 for _ in range(8)]

        for i in range(8):
            if self.visited[self.pos[0]][self.pos[1]]%(2**i) == 0:
                actions[i] = -100
        return actions

