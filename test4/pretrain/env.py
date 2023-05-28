from env_base import EnvironmentBase

class Environment(EnvironmentBase):
    def action(self, action):
        super().action(action)

        flag0 = False
        flag1 = False
        for x, y in self.gate0:
            if x == self.pos[0] and y == self.pos[1]:
                flag0 = True
                break
        for x, y in self.gate1:
            if x == self.pos[0] and y == self.pos[1]:
                flag1 = True
                break        
        done = (self.visited[self.pos[0]][self.pos[1]] == 0 or flag1 or flag0)
        
        return done
    
    def get_state(self):
        state = []
        for i in range(self.width*self.height):
            for j in range(8):
                state.append((self.visited[i//self.height][i%self.height]//(2**j))%2)

        for i in range(self.width*self.height):
            state.append(0 if self.pos[0] != i//self.height and self.pos[1] != i%self.height else 1)
        state.append(self.player)
        return state, self.get_norm()
    
    def get_norm(self):
        actions = [1 for _ in range(8)]

        for i in range(8):
            if (self.visited[self.pos[0]][self.pos[1]]//(2**i))%2 == 0:
                actions[i] = -100
        return actions

