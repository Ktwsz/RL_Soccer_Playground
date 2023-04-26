from Environment.base import EnvironmentBase
import pygame

class Environment(EnvironmentBase):
    def action(self, action):
        super().action(action)

        if self.pos in self.gate0:
            done = 1
            if self.old_player == 0:
                reward = -1
            else:
                reward = 1
        elif self.pos in self.gate1:
            done = 1
            if self.old_player == 0:
                reward = 1
            else:
                reward = -1
        else:
            done = 0
            reward = 0
        print(action)
        pygame.image.save(self.screen,f"screenshots/shot.jpg")#{self.counter}
        self.counter += 1
        return done, reward