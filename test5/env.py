from env_base import EnvironmentBase
import numpy as np
from torchvision import transforms

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
        
        if self.pos == self.old_pos:
            done = 0
            reward = -100
        elif self.visited[self.pos[0]][self.pos[1]] == 0:
            done = 1
            reward = -100
        elif flag0:
            done = 1
            if self.old_player == 0:
                reward = 0
            else:
                reward = 1
        elif flag1:
            done = 1
            if self.old_player == 0:
                reward = 1
            else:
                reward = 0
        else:
            done = 0
            reward = 0

        return reward, self.get_state(), done
    
    def get_state(self):

        trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

        state = np.ndarray(shape=((self.height-1)*3+1, (self.width-1)*3+1, 3), dtype=np.uint8)
        for i in range(3*(self.height-1)+1):
            for j in range(3*(self.width-1)+1):
                state[i][j] = np.array(self.state[j][i], dtype=np.uint8)
        
        return trans(state)

