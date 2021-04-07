
#####################################################################
# Flappy Bird                                                       #
# Creator: Vikky Panchal                                            #
# This is a simple flappy game, which uses neuro-evolution to play  #
# Flappy Bird Game.                                                 #
# Using pygame                                                      #
#####################################################################

import os, sys, pygame
from pygame.locals import *
import numpy as np
from nn import neural_network as nn

screen_width, screen_height = 1400, 800
FPS = 60

pipe_width = 100
pipe_horizontal_gap = 300
pipe_vertical_gap = 200
pipe_count = (screen_width//(pipe_horizontal_gap + pipe_width)) + 2

###############################################################
# Parameter's that you can tweek, which change the resulting  #
# model learning rate                                         #
###############################################################
bird_population = 100
bird_speed = 10
pipe_speed = 6
do_not_mutate_percentage = 0.4
mutate_percentage = 0.6
hidden_nodes = 20


#############################################################
# Color class                                               #
#############################################################
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

input_nodes, output_nodes = 3, 1
sizes=[input_nodes, hidden_nodes, output_nodes]

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

####################################################
# Bird Class                                       #
# Holds: x location, y location, velocity,         #
# brain (instance of neural_network),              #
# dead (flag to check if bird is alive             #
#################################################### 
class Bird(pygame.sprite.Sprite):
    def __init__(self, x_loc, y_loc, velocity, brain=None):
        super(Bird, self).__init__()
        #pygame.sprite.Sprite.__init__(self)

        self.velocity = velocity
        self.x_loc = x_loc
        self.y_loc = y_loc

        self.image, self.rect = load_image("bird.png", (40, 40), Color.WHITE)
        self.rect.center = (x_loc, y_loc)
        if brain is None:
            self.brain = nn(sizes)
        else:
            self.brain = brain
        self.bird_dead = False

    def boundary_collition(self):
        if self.rect.bottom>=screen_height or self.rect.top<=0:
            return True

    def evaluate(self, input):
        #input = np.array(input)
        #return self.brain.predict(np.reshape(input, (3,1)))
        return self.brain.predict(input)

    def update(self):
        self.rect.y -= self.velocity
        self.velocity = self.velocity-1

    def jump(self):
        self.velocity = 10

    def vel(self):
        return bird_speed

    def get_X(self):
        return self.rect.x

    def get_brain(self):
        return self.brain

    def get_Y(self):
        return self.rect.y

    def set_dead(self):
        self.bird_dead = True

    def isDead(self):
        return self.bird_dead

#######################################################
# Upper pipe class                                    # 
#######################################################
class Pipe_Upper(pygame.sprite.Sprite):
    def __init__(self, x, height, speed):
        super(Pipe_Upper, self).__init__()
        #pygame.sprite.Sprite.__init__(self)

        self.pipe_speed = speed
        self.pipe_height = height
        #self.image = pygame.Surface((pipe_width, height))
        #self.image.fill(Color.GREEN)
        #self.image.set_colorkey(Color.WHITE)
        #self.rect = self.image.get_rect()
        self.image, self.rect = load_image("pipe.png", (int(pipe_width), int(height)), Color.WHITE, float(180))
        self.rect.x = (x)
        self.rect.y = (0)
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect.x -= self.pipe_speed

    def get_X(self):
       return self.rect.x

    def get_Y(self):
        return (self.rect.y + self.pipe_height)

#######################################################
# Lower pipe class                                    #
#######################################################
class Pipe_Lower(pygame.sprite.Sprite):
    def __init__(self, x, height, speed):
        super(Pipe_Lower, self).__init__()
        #pygame.sprite.Sprite.__init__(self)

        self.pipe_speed = speed
        self.image, self.rect = load_image("pipe.png", (int(pipe_width), int(screen_height - (pipe_vertical_gap + height))), Color.WHITE)
        #self.image = pygame.Surface((pipe_width, screen_height - (pipe_vertical_gap + height)))
        #self.image.fill(Color.GREEN)
        #self.image.set_colorkey(Color.WHITE)
        #self.rect = self.image.get_rect()
        self.rect.x = (x)
        self.rect.y = (pipe_vertical_gap + height)
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect.x -= self.pipe_speed

    def get_X(self):
       return self.rect.x

    def get_Y(self):
        return self.rect.y

def init_birds(bird_sprites, brain):
    bird_brain_list=[None]*bird_population
    if brain is not None:
        # logic to mutate brid
        bird_40_percent = [brain]*int(bird_population*do_not_mutate_percentage)
        bird_60_percent = [brain]
        for i in range(int((bird_population*mutate_percentage) - 1)):
            b = brain.copy()
            b.mutate()
            bird_60_percent.append(b)
        bird_brain_list = bird_40_percent + bird_60_percent

    bird_list=[]
    x = screen_width//8
    y = np.random.randint(low = 0,high=screen_height,size=bird_population)
    for i in range(bird_population):
        bird = Bird(x, y[i], bird_speed, bird_brain_list[i])
        bird_list.append(bird)
        bird_sprites.add(bird)
    return bird_list

def init_pipes(pipe_sprites):
    pipe_list = []
    start_x = 500
    for pipe_index in range(pipe_count):
        pipe_height = (round(np.random.uniform(0.15,0.85), 2))*(screen_height-pipe_vertical_gap)
        pipe_upper = Pipe_Upper(start_x, pipe_height, pipe_speed)
        pipe_lower = Pipe_Lower(start_x, pipe_height, pipe_speed)
        start_x += pipe_width + pipe_horizontal_gap
        pipe_sprites.add(pipe_upper)
        pipe_sprites.add(pipe_lower)
        pipe_list.append([pipe_upper, pipe_lower])
    return pipe_list

def make_pipe(pipe):
    pipe_height = (round(np.random.uniform(0.15,0.85), 2))*(screen_height-pipe_vertical_gap)
    pipe_upper = Pipe_Upper(pipe[0].get_X()+pipe_width+pipe_horizontal_gap, pipe_height, pipe_speed)
    pipe_lower = Pipe_Lower(pipe[0].get_X()+pipe_width+pipe_horizontal_gap, pipe_height, pipe_speed)
    return [pipe_upper, pipe_lower]


######################################################
# Loop that finds the best bird, from the population #
# and return's to repopulate the next generation     #
######################################################
def loop(generation, score, bird_list, pipe_list, nearest_pipe, bird_sprites, pipe_sprites):
    myfont = pygame.font.SysFont("monospace", 16)
    run_score, bird_count, best_index = 0, 0, -1
    best_bias, best_score, best_bird_index = [], [], []

    while True:
        clock.tick(FPS)
        screen.fill(Color.WHITE)

        screen_size = screen.get_size()
        background, _ = load_image("background.png", (screen_width, screen_height))
        screen.blit(background, (0, 0))

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        # Iterate over all the birds.
        for bird_index in range(bird_population):
            bird = bird_list[bird_index]
            if bird.isDead() == False:
                input = []

               # Get Birds x and y locations.
                bird_x, bird_y = bird.rect.center

               # Get pipe x location.
                pipe_x = nearest_pipe[0].get_X()
                draw_x,draw_y = 0,0

                ypip = (nearest_pipe[0].get_Y()+(pipe_vertical_gap//2))/(screen_height)
                draw_y = nearest_pipe[0].get_Y()+(pipe_vertical_gap//2)

                ####################################################
                # This condition checks if bird is before pipe     #
                ####################################################
                if bird_x <= pipe_x:
                    xpip = (nearest_pipe[0].get_X())/(screen_width)
                    draw_x = nearest_pipe[0].get_X()
                ####################################################
                # This condition checks if bird is inside pipe     #
                ####################################################
                elif bird_x <= (pipe_x + (pipe_width//2)):
                    xpip = (nearest_pipe[0].get_X()+(pipe_width//2))/(screen_width)
                    draw_x = nearest_pipe[0].get_X() + (pipe_width//2)
                ####################################################
                # This condition checks if bird is past the pipe   #
                ####################################################
                else:
                    xpip = (nearest_pipe[0].get_X()+pipe_width)/(screen_width)
                    draw_x = nearest_pipe[0].get_X() + pipe_width

                pygame.draw.circle(screen,Color.RED,(draw_x,int(draw_y)),5)
                pygame.draw.line(screen, Color.YELLOW, (bird_x, bird_y), (draw_x,draw_y))

                ###################################################
                # The input to neural network is :                #
                # bird speed,                                     #
                # delta_x : difference between pipe_x and bird_x, #
                # delta_y : difference between pipe_y and bird_y. #
                ###################################################
                input.append(bird.vel())
                input.append((bird_x/screen_width)-xpip)
                input.append((bird_y/screen_height)-ypip)
                direct_distance = np.sqrt(((input[1]**2) + (input[2]**2)))
                fitness = run_score - direct_distance
                output = bird.evaluate(input)

                #####################################################
                # If the model outputs value greater then 0.5, then #
                # our bird jumps.                                   #
                #####################################################
                if output[0] > 0.5:
                    bird.jump()

                #####################################################
                # Here we check if bird collides with screen        #
                # boundary, if it does, we remove the bird and make #
                # a note of it's fitness score                      #
                #####################################################
                if bird.boundary_collition():
                    #print ('Collision: bird_index:{}, x:{}, y:{}, bottom:{}, top:{}'.format(bird_index, bird_x, bird_y, bird.rect.bottom, bird.rect.top))
                    pygame.sprite.Sprite.kill(bird)
                    best_score.append(fitness)
                    best_bird_index.append(bird_index)
                    bird.set_dead()
                    bird_count += 1

                #####################################################
                # Here we check if bird collides with pipe          #
                # boundary, if it does, we remove the bird and make #
                # a note of it's fitness score                      #
                #####################################################
                for pipe in nearest_pipe:
                    c = 0
                    if pygame.sprite.collide_rect(bird,pipe) and (bird.isDead() == False):
                        bird_hits = pygame.sprite.spritecollide(bird,pipe_sprites,False,pygame.sprite.collide_mask)
                        if bird_hits:
                            c = 1
                            pygame.sprite.Sprite.kill(bird)
                            bird.set_dead()
                            best_score.append(fitness)
                            best_bird_index.append(bird_index)
                            bird_count += 1
                            break
                    if c == 1:
                        break

            #####################################################
            # When all birds have died, we pick the bird with   #
            # maximum fitness, and use it to mutate and spawn   #
            # next generation of birds                          # 
            #####################################################
            if bird_count == len(bird_list):
                if max(best_score) > score:
                    score = max(best_score)
                    try:
                        best_index = list(best_score).index(max(best_score))
                        best_index = best_bird_index[best_index]
                    except:
                        print(best_score)
                        print(best_bird_index)
                        sys.exit(0)
                birds_alive = myfont.render("Birds Alive {0}".format(len(bird_list)-bird_count), 1, (0,0,0))
                update_screen_scores(screen, score, run_score, birds_alive)
                return best_index,score

        bird_sprites.update()
        pipe_sprites.update()
        bird_sprites.draw(screen)
        pipe_sprites.draw(screen)

        ######################################################
        # If the pipe goes off screen, we remove it, and add #
        # new pipe to the pipe_list                          #
        ######################################################
        if pipe_list[0][0].get_X() + pipe_width <= 0:
            pygame.sprite.Sprite.kill(pipe_list[0][0])
            pygame.sprite.Sprite.kill(pipe_list[0][1])
            pipe_list.pop(0)
            pipe_list.append(make_pipe(pipe_list[-1]))
            pipe_sprites.add(pipe_list[-1][0])
            pipe_sprites.add(pipe_list[-1][1])

        ###################################################
        # we need to update the next nearest pipe. The way#
        # we do this, when bird passes the pipe+width     #
        # screen_width//8 : Bird X location               #
        ###################################################
        if nearest_pipe[0].get_X() + pipe_width <= (screen_width//8):
            nearest_pipe = pipe_list[1]

        birds_alive = myfont.render("Birds Alive {0}".format(len(bird_list)-bird_count), 1, (0,0,0))
        update_screen_scores(screen, score, run_score, birds_alive)
        run_score += 1

def update_screen_scores(screen, score, run_score, birds_alive):
    myfont = pygame.font.SysFont("monospace", 16)
    gen = myfont.render("Genertation {0}".format(generation), 1, (0,0,0))
    highest = myfont.render("Highest Score {0}".format(int(round(score))), 1, (0,0,0))
    current = myfont.render("Current Score {0}".format(run_score), 1, (0,0,0))
    screen.blit(gen, (5, 10))
    screen.blit(highest, (5, 35))
    screen.blit(current, (5, 60))
    screen.blit(birds_alive, (5, 85))
    pygame.display.flip()

def purge_all():
    print ("PURGE ALL")
    pass

if __name__ == "__main__":
    print('Game Start')
    old_brain = None
    brain = None
    score = 0
    threshold_score = 0
    threshold_count = 0
    generation = 1
    np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
    while True:
        pygame.init()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Flappy Bird')

        bird_sprites, pipe_sprites = pygame.sprite.Group(), pygame.sprite.Group()

        #####################################################
        # Create bird population                            #
        #####################################################
        bird_list = init_birds(bird_sprites, brain)

        #####################################################
        # Draw pipes                                        #
        #####################################################
        pipe_list = init_pipes(pipe_sprites)

        nearest_pipe = pipe_list[0]
        if brain is not None:
            old_brain = brain
        index, score = loop(generation, score, bird_list, pipe_list, nearest_pipe, bird_sprites, pipe_sprites)
        if index == -1:
            pygame.quit()
        
        generation += 1
        if threshold_score < score:
            threshold_score = score
            threshold_count = 0
        else:
            threshold_count += 1

        if threshold_count >= 5:
            purge_all()
            threshold_count = 0
            brain = None
            old_brain = None
        else:
            print('It time to mutate best bird')
            if index == -1:
                brain = old_brain
            else:    
                brain = bird_list[index].get_brain()
