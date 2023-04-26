import pygame
from copy import copy

class EnvironmentBase(object):
    def __init__(self, width, height):
        pygame.init()

        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode(((self.width-1)*50+20, (self.height-1)*50))
        self.screen.fill((255,255,255))
        
        self.reset()

        self.rev_action = [1, 0, 3, 2, 7, 6, 5, 4]

        self.gate0 = [(self.pos[0], 0), (self.pos[0]-1, 0), (self.pos[0]+1, 0)]
        self.gate1 = [(self.pos[0], self.height-1), (self.pos[0]-1, self.height-1), (self.pos[0]+1, self.height-1)]
        #print(self.gate0, self.gate1)

    def draw_env(self):
        for i in range(self.width):
            for j in range(self.height):
                pygame.draw.circle(self.screen, (0,0,0), (22+i*45,10+j*45), 5)
        player_color = (255,0,0) if self.player == 0 else (0,0,255)
        pygame.draw.circle(self.screen, player_color, (22+self.pos[0]*45,10+self.pos[1]*45), 5)
        pygame.display.update()

    def draw_line(self, target):
        player_color = (255,0,0) if self.player == 0 else (0,0,255)
        pygame.draw.circle(self.screen, (0,0,0), (22+target[0]*45,10+target[1]*45), 5)
        pygame.draw.line(self.screen, (0,0,0), (22+45*self.pos[0], 10+self.pos[1]*45), (22+target[0]*45, 10+target[1]*45), 3)
        pygame.draw.circle(self.screen, player_color, (22+self.pos[0]*45,10+self.pos[1]*45), 5)
        pygame.display.update()

    def action(self, action):
        self.old_pos = copy(self.pos)
        self.old_player = copy(self.player)
        if (self.visited[self.pos[0]][self.pos[1]] // (2**action)) % 2 == 1:
            #print("from", self.pos[0], self.pos[1], self.visited[self.pos[0]][self.pos[1]])
            self.visited[self.pos[0]][self.pos[1]] -= 2**action
            match action:
                case 0:
                    if self.pos[0]-1 >= 0:
                        self.pos[0] -= 1
                case 1:
                    if self.pos[0]+1 < self.width:
                        self.pos[0] += 1
                case 2:
                    if self.pos[1]-1 >= 0:
                        self.pos[1] -= 1
                case 3:
                    if self.pos[1]+1 < self.height:
                        self.pos[1] += 1
                case 4:
                    if self.pos[0]-1 >= 0 and self.pos[1]-1 >= 0:
                        self.pos[0] -= 1
                        self.pos[1] -= 1
                case 5:
                    if self.pos[0]-1 >= 0 and self.pos[1]+1 < self.height:
                        self.pos[0] -= 1
                        self.pos[1] += 1
                case 6:
                    if self.pos[0]+1 < self.width and self.pos[1]-1 >= 0:
                        self.pos[0] += 1
                        self.pos[1] -= 1
                case 7:
                    if self.pos[0]+1 < self.width and self.pos[1]+1 < self.height:
                        self.pos[0] += 1
                        self.pos[1] += 1

            if self.old_pos[0] != self.pos[0] or self.old_pos[1] != self.pos[1]:
                self.visited[self.pos[0]][self.pos[1]] -= 2**self.rev_action[action]

            self.draw_line(self.old_pos)
            if not self.vertices_visited[self.pos[0]][self.pos[1]]:
                self.vertices_visited[self.pos[0]][self.pos[1]] = True
                self.player = 1-self.player
            #print("to", self.pos[0], self.pos[1], self.visited[self.pos[0]][self.pos[1]])
            
    def reset(self):
        self.pos = [self.width//2, self.height//2]

        self.screen.fill((255,255,255))

        self.visited = [ [ 255 for i in range(self.height) ] for j in range(self.width) ]
        self.vertices_visited = [ [ False for i in range(self.height) ] for j in range(self.width) ]
        self.vertices_visited[self.pos[0]][self.pos[1]] = True

        self.player = 0      
        
        self.counter = 0
        
        self.draw_env()
