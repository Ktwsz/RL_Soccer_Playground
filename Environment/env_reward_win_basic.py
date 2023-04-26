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
            reward = -1
        elif flag0:
            done = 1
            if self.old_player == 0:
                reward = -1
            else:
                reward = 1
        elif flag1:
            done = 1
            if self.old_player == 0:
                reward = 1
            else:
                reward = -1
        else:
            done = 0
            reward = 0

        return reward, self.get_state(), done
    
    def get_state(self):
        state = []
        for i in range(self.width*self.height):
            state.append(1 if self.vertices_visited[i//self.height][i%self.height] else 0 )

        state.append(self.pos[0])
        state.append(self.pos[1])
        for i in range(8):
            state.append((self.visited[self.pos[0]][self.pos[1]]//(2**i))%2)
        state.append(self.player)
        return state

