from Environment.base import EnvironmentBase

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

        if self.visited[self.pos[0]][self.pos[1]] == 0:
            done = 1
            reward = -100
        elif flag0:
            done = 1
            if self.old_player == 0:
                reward = -100
            else:
                reward = 100
        elif flag1:
            done = 1
            if self.old_player == 0:
                reward = 100
            else:
                reward = -100
        elif self.vertices_visited[self.pos[0]][self.pos[1]]:
            reward = 1
            done = 0
        elif self.pos[0]-self.old_pos[0] > 0:
            if self.old_player == 0:
                reward = -10
            else:
                reward = 10
            done = 0
        elif self.pos[0]-self.old_pos < 0:
            if self.old_player == 0:
                reward = 10
            else:
                reward = -10
            done = 0
        else:
            reward = -100
            done = 0

        return reward, self.get_state(), done
    
    def get_state(self):
        state = []
        for i in range(self.width*self.height):
            for j in range(8):
                state.append((self.visited[i//self.height][i%self.height]//(2**j))%2)

        for i in range(self.width*self.height):
            state.append(0 if self.pos[0] != i//self.height and self.pos[1] != i%self.height else 1)
        state.append(self.player)
        return state

