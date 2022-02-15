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
from generator import Generator

ale = ALEInterface()
ale.loadROM('river-raid.bin')
legal_actions=ale.getLegalActionSet()

ale.reset_game()

if torch.cuda.is_available():
    print('cuda is available')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device).float()
generator.load_state_dict(torch.load('generator.pt',map_location=torch.device('cpu')))
generator.eval()

def run():
    sdl2.ext.init()
    window = sdl2.ext.Window("Prediction", size=(128,128))
    window.show()

    ground_window = sdl2.ext.Window("Ground truth", size=(128,128))
    ground_window.show()

    running = True
    action = Action.NOOP
    while running:
        #sleep(0.01)
        ale.act(action)

        if ale.game_over():
            ale.reset_game()

        groundsurface = ground_window.get_surface()
        pixelview = sdl2.ext.pixels3d(groundsurface)
        pix = ale.getScreenRGB()
        pix = Image.fromarray(pix)
        pix = torchvision.transforms.Resize( (ground_window.size[1], ground_window.size[0]),
                                             torchvision.transforms.InterpolationMode.NEAREST ).forward( pix )
        pix = np.array(pix).swapaxes(0,1)
        pixelview[:, :, [2,1,0]] = pix

        ground_window.refresh()
        
        windowsurface = window.get_surface()
        pixelview = sdl2.ext.pixels3d(windowsurface)

        memory = ale.getRAM()
        memory = np.reshape( np.unpackbits(memory), (1, 1, 32,32) )
        memory = memory.astype(float)
        memory = torch.from_numpy(memory).float()
        pix = generator(memory)
        x = pix.cpu().detach().numpy()[0]
        x = x.transpose((1, 2, 0))
        x = (x * 255).astype(np.uint8)
        pix = Image.fromarray(x)
        pix = torchvision.transforms.Resize( (window.size[1], window.size[0]),
                                             torchvision.transforms.InterpolationMode.NEAREST ).forward( pix )
        pix = np.array(pix).swapaxes(0,1)
        pixelview[:, :, [2,1,0]] = pix

        window.refresh()

        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_KEYDOWN:
                if event.key.keysym.sym == sdl2.SDLK_UP:
                    action = Action.FIRE
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
