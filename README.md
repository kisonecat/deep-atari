# deep-atari

Watch the YouTube demo at https://youtu.be/UGFrE6eHYfw

Using `learn.py` which is based on

https://learnopencv.com/paired-image-to-image-translation-pix2pix/

a neural network is trained to predict the visual output of an Atari
2600 given the contents of its RAM.  Then `play.py` lets you play the
resulting game while viewing the predicted output.

## Credits

The GAN code is modified from

https://learnopencv.com/paired-image-to-image-translation-pix2pix/

and uses [PyTorch](https://github.com/pytorch/pytorch).

I used the [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) to emulate the Atari 2600.
