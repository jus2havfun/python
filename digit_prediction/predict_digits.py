import os, sys, random
import pygame as py
from PIL import Image
from pygame.locals import *
from matplotlib import pyplot as plt
from nn import neural_network, normalize
import numpy as np

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


##########################################
# Function to add test to screen         #
##########################################
def put_text(surface, xloc, yloc, color, txt):
    myfont = py.font.SysFont("monospace", 20)
    start_pos = myfont.render(txt, 1, color)
    surface.blit(start_pos, (xloc, yloc))

##########################################
# Load the saved png image into memory   #
##########################################
def load_image(filename):
    img = py.image.load(filename)
    x = py.image.tostring(img, "RGB", True)
    img = Image.frombytes("RGB", img.get_size(), x).convert("L")
    x = img.tobytes('raw', 'L', 0, -1)
    X = [int(i) for i in x]
    return X


#########################################
# This is the main loop which captures  #
# the image draw as a png file on disk. #
# which is later read by the code as    #
# bytes and passed to my simple neural  #
# network which was loaded by pretrained#
# model to make predictions             #
#########################################
cnt = 0
def loop():
    running = True
    global cnt
    while running:
        screen.fill(Color.BLACK)
        py.display.flip()
        for event in py.event.get():
            if event.type == QUIT:
                py.quit()
                quit()
            elif event.type == py.MOUSEBUTTONDOWN:
                running1 = True
                while running1:
                    for event in py.event.get():
                        if event.type == py.MOUSEMOTION:
                            py.draw.circle(screen, Color.WHITE, py.mouse.get_pos(), 40)
                            py.display.flip()
                        elif event.type == py.MOUSEBUTTONUP:
                            cnt += 1
                            img = py.transform.scale(screen, (28, 28))
                            filename = "save\{}.png".format(cnt)
                            #py.image.save(img, filename)
                            py.image.save(img, "img.png")
                            running1 = False
                            running = False
                            break

if __name__ == "__main__":
    py.init()
    sizes = [784, 128, 64, 32, 10]

    ############################################
    # Create a neural_network and load with    #
    # pretrained model file                    #
    ############################################
    model = neural_network(sizes)
    model.load("C:\Vikky\Documents\GitHub\python\prediction\\nn_digits_784_128_64_32_10.dat")
    screen = py.display.set_mode((700, 700))

    correct, wrong = 0, 0
    while True:
        loop()
        X = load_image("img.png")
        X = normalize(np.reshape(X, (784, 1)))
        p = model.predict(X)
        s = str("Predicted digit:{}, accuracy:{}".format(p, correct/(correct+wrong+0.00001)))
        plt.imshow(np.reshape(X, (28,28)), cmap='gray_r')
        plt.text(0, -2, s)
        plt.show()
