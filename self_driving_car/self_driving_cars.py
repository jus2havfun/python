import pygame, random, math, sys, os
from pygame.sprite import Group, Sprite
import numpy as np
from pygame.locals import *
from nn import neural_network as nn

screen_width, screen_height = 0, 0
map, maprect = None, None
FPS = 60

########################################
# Parameters you can tweek             #
########################################
angle = 0
no_of_players = 4
do_not_mutate_percentage = 0.4
mutate_percentage = 0.6
car_start_pos = (179, 330)
clock = None

class Color:
    RED = (255, 0, 0)
    GREEN = (0, 100, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)
    GRAY = (127, 127, 127)
    WHITE = (255, 255, 255)
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)

#####################################
# Class for player/car.             #
#####################################
class Player(Sprite):
    def __init__(self, image, position, anchor, map, angle, brain = None):
        Sprite.__init__(self)
        self.o_image = image
        self.image = image
        self.map = map
        self.rect = image.get_rect(**{anchor: position})
        self.center = pygame.Vector2(self.rect.center)

        sizes = [5, 10, 2]
        if brain is None:
            self.brain = nn(sizes)
        else:
            self.brain = brain

        self.dead = False

        self.motion = 2 
        self.rotate_angle = 0
        self.angle = angle
        self.img_corners = []
        self.radar = [None]*5
        self.input = []
        self.distance = 0

        tmp = 0
        for i in range(5):
            self.radar[i] = pygame.Vector2()
            self.radar[i].from_polar((1, 360 - (self.angle + tmp)))
            tmp += 45

        self.compute_corners(angle)
        self.compute_radars()

    def get_brain(self):
        return self.brain

    def evaluate(self):
        input = []
        for i in range(len(self.input)):
            input.append(self.input[i][1]/100)
        return self.brain.predict(np.reshape(input, (5,1)))

    def update(self, surface):
        if self.dead == True:
            return
        self.distance += 1
        if self.motion != 0:
            self.center += self.radar[2] * self.motion
            self.rect.center = self.center

        if self.rotate_angle != 0:
            self.rotate(self.angle + self.rotate_angle, surface)
            self.rotate_angle = 0

        if self.dead == False:
            self.compute_corners(self.angle)
            self.compute_radars()

    def is_alive(self):
        return self.dead == False

    def rotate(self, angle, surface):
        self.angle = angle
        # reverse rotation
        tmp = 0
        for i in range(5):
            self.radar[i].from_polar((1, 360 - (self.angle +tmp)))
            tmp += 45
        self.image = pygame.transform.rotate(self.o_image, angle)
        self.rect = self.image.get_rect(center=self.rect.center)

    ################################################
    # Method that detects race track collision     #
    ################################################
    def collide(self):
        for corner in self.img_corners:
            if self.map.get_at(corner) == (126, 126, 126, 255):
                self.dead = True
                return True
        return False

    ################################################
    #  Method that computes radar lengths          #
    ################################################
    def compute_radars(self):
        self.input = []
        for i in range(0, 5):
            b = None
            len = 0
            b = self.center + self.radar[i] * len
            while not self.map.get_at((int(b.x), int(b.y))) == (126, 126, 126, 255) and len <= 100:
                len += 4
                b = self.center + self.radar[i] * len
            self.input.append([b, len])
        
    #############################################
    # Compute car corners as it manovers through#
    # the race track.                           #
    #############################################
    def compute_corners(self, angle):
        len =  23
        self.img_corners = []
        self.img_corners.append((int(self.rect.center[0]+ math.cos(math.radians(360 - (angle + 45))) * len), int(self.center[1] + math.sin(math.radians(360 - (angle + 45))) * len)))
        self.img_corners.append((int(self.rect.center[0]+ math.cos(math.radians(360 - (angle+135))) * len), int(self.center[1] + math.sin(math.radians(360 - (angle + 135))) * len)))
        self.img_corners.append((int(self.rect.center[0]+ math.cos(math.radians(360 - (angle+225))) * len), int(self.center[1] + math.sin(math.radians(360 - (angle + 225))) * len)))
        self.img_corners.append((int(self.rect.center[0]+ math.cos(math.radians(360 - (angle+315))) * len), int(self.center[1] + math.sin(math.radians(360 - (angle + 315))) * len)))

def create_player_image(angle):
    image = pygame.image.load("images\car.png").convert()
    image = pygame.transform.scale(image, (30, 30))
    image = pygame.transform.rotate(image, angle)
    image.set_colorkey((0,0,0))
    return image

def load_image(name, scale, colorkey=None, rotate=None):
    fullname = os.path.join(os.path.dirname(__file__) + '/images', name)
    try:
        image = pygame.image.load(fullname)
    except pygame.error as message:
        print('Cannot load image:', name)
        raise SystemExit(message)
    image = image.convert()
    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey)
    image = pygame.transform.scale(image, scale)
    if rotate is not None:
        image = pygame.transform.rotate(image, rotate)
    return image, image.get_rect()


def load_map():
    return load_image("map1.png", (screen_width, screen_height))

def loop(generation, score, screen, players_list, players_sprites, map, maprect):
    player_count, run_score, best_index  = 0, 0, -1
    best_score, best_player_index = [], []
    print (screen_width, screen_height)
    myfont = pygame.font.SysFont("monospace", 16)
    while True:
        clock.tick(FPS)
        screen.fill(Color.BLACK)
        screen.blit(map, (0, 0))
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        for player_index in range(no_of_players):
            player = players_list[player_index]
            if player.is_alive() :
                output = player.evaluate()
                if output[0] == 0:
                    player.rotate_angle += 2
                else:
                    player.rotate_angle -= 2

                if player.collide():
                    pygame.sprite.Sprite.kill(player)
                    best_score.append(player.distance)
                    best_player_index.append(player_index)
                    player.dead = True
                    player_count += 1

                if player.is_alive() :
                    a = player.center
                    for input in player.input:
                        b = input[0]
                        pygame.draw.line(screen, Color.RED, (int(a.x), int(a.y)), (int(b.x), (int(b.y))))
                        point2 = myfont.render("{}".format((int(b.x), int(b.y))), 1, (255,255,255))
                        screen.blit(point2, (int(b.x), int(b.y)))

        if player_count == len(players_list):
            print('All players died')
            if max(best_score) > score:
                score = max(best_score)
                try:
                    best_index = list(best_score).index(max(best_score))
                    best_index = best_player_index[best_index]
                except:
                    print(best_score)
                    print(best_player_index)
                    sys.exit(0)    
            players_alive = myfont.render('Players Alive {0}'.format(len(players_list) - player_count), 1, (0, 0, 0))
            update_scores(screen, score, run_score, players_alive)
            pygame.display.flip()
            return best_index, score

        players_sprites.update(screen)
        players_sprites.draw(screen)
        players_alive = myfont.render('Players Alive {0}'.format(len(players_list) - player_count), 1, (0, 0, 0))
        update_scores(screen, score, run_score, players_alive)
        pygame.display.flip()
        run_score += 1

def update_scores(screen, score, run_score, players_alive):
    myfont = pygame.font.SysFont("monospace", 16)
    gen = myfont.render("Genertation {0}".format(generation), 1, (0,0,0))
    highest = myfont.render("Highest Score {0}".format(int(round(score))), 1, (0,0,0))
    current = myfont.render("Current Score {0}".format(run_score), 1, (0,0,0))
    screen.blit(gen, (5, 10))
    screen.blit(highest, (5, 35))
    screen.blit(current, (5, 60))
    screen.blit(players_alive, (5, 85))

def init_players(players_sprites, screen, map, brain = None):
    player_brain_list = [None]*no_of_players
    if brain is not None:
        player_40_percent = [brain]*round(no_of_players*do_not_mutate_percentage)
        player_60_percent = [brain]
        for i in range(round((no_of_players*mutate_percentage) - 1)):
            b = brain.copy()
            b.mutate()
            player_60_percent.append(b)
        player_brain_list = player_40_percent + player_60_percent

    players_list = []
    for i in range(no_of_players):
        image = create_player_image(angle)
        player = Player(image, car_start_pos, "center", map, angle, player_brain_list[i])
        players_list.append(player)
        players_sprites.add(player)
    return players_list

if __name__ == "__main__":
    generation = 1
    score = 0
    threshold_score = 0
    threshold_count = 0
    brain, old_brain = None, None
    while True:
        pygame.init()
        clock = pygame.time.Clock()
        infoObject = pygame.display.Info()
        screen_width, screen_height = infoObject.current_w, infoObject.current_h;
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Self Driving Cars in Race Track')
        players_sprites = pygame.sprite.Group()

        map, maprect = load_map()
        players_list = init_players(players_sprites, screen, map, brain)
        if brain is not None:
            old_brain = brain

        index, score = loop(generation, score, screen, players_list, players_sprites, map, maprect)
        if index == -1:
            pygame.quit()

        generation += 1

        if threshold_score < score:
            threshold_score = score
            threshold_count = 0
        else:
            threshold_count += 1

        if threshold_count >= 5:
            print ('PURGE ALL')
            threshold_count = 0
            brain = None
            old_brain = None
        else:
            print('It time to mutate best car')
            if index == -1:
                brain = old_brain
            else:    
                brain = players_list[index].get_brain()
