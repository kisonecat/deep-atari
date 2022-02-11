import sys
import sdl2
import sdl2.ext
from ale_py import ALEInterface, Action
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
from time import sleep
ale = ALEInterface()
ale.loadROM('river-raid.bin')
legal_actions=ale.getLegalActionSet()

ale.reset_game()

#        self.ale.act( self.legal_actions[randrange(len(self.legal_actions))] )


def run():
    sdl2.ext.init()
    window = sdl2.ext.Window("The Pong Game", size=(128,128))
    window.show()
    windowsurface = window.get_surface()
    pixelview = sdl2.ext.pixels3d(windowsurface)

    #with Image.open("output.png") as im:
    #    pix = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
    #    pix = pix.swapaxes(0,1)
    #    pixelview[:, :, [2,1,0]] = pix
    #    window.refresh()

    running = True
    action = Action.NOOP
    while running:
        sleep(0.01)
        ale.act(action)

        if ale.game_over():
            ale.reset_game()
        
        windowsurface = window.get_surface()
        pixelview = sdl2.ext.pixels3d(windowsurface)
        pix = ale.getScreenRGB()
        pix = Image.fromarray(pix)
        pix = torchvision.transforms.Resize( (window.size[1], window.size[0]),
                                             torchvision.transforms.InterpolationMode.NEAREST ).forward( pix )
        pix = np.array(pix).swapaxes(0,1)
        pixelview[:, :, [2,1,0]] = pix

        window.refresh()

        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_KEYDOWN:
                if event.key.keysym.sym == sdl2.SDLK_UP:
                    action = Action.UP
                elif event.key.keysym.sym == sdl2.SDLK_DOWN:
                    action = Action.DOWN
                elif event.key.keysym.sym == sdl2.SDLK_RIGHT:
                    action = Action.RIGHT
                elif event.key.keysym.sym == sdl2.SDLK_LEFT:
                    action = Action.LEFT
                elif event.key.keysym.sym == sdl2.SDLK_RETURN:
                    action = Action.FIRE
            if event.type == sdl2.SDL_KEYUP:
                action = Action.NOOP
                    
            if event.type == sdl2.SDL_QUIT:
                running = False
                break

    return 0

if __name__ == "__main__":
    sys.exit(run())
