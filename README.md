# Image Generator

This is an AI Image Generator, similar to (although obviously not nearly at the level of) img2img functionality of Stable Diffusion, as well as many other AI Image generator/editors.

The program takes in a sketch as input, and returns a photograph, based on the data used for training.

The current dataset I am using to train it is [BIRDS 525 SPECIES](https://www.kaggle.com/datasets/gpiosenka/100-bird-species) from kaggle, I trained a model with the 150+ images of an "Abbots Babbler", as well as 10 different "sketch" versions of each image that i had automatically created using my [Simplify](https://github.com/EgeEken/Simplify) algorithm

## INPUT
<img width="239" alt="image" src="https://github.com/EgeEken/Image-Generator/assets/96302110/9ab49256-bea7-4408-8847-4016fe5a1bac">

## TRAINING IMAGES
<img width="1124" alt="image" src="https://github.com/EgeEken/Image-Generator/assets/96302110/ce915697-dac4-465d-93f7-6ea823d2ce32">

## EXPECTED OUTPUT
<img width="240" alt="image" src="https://github.com/EgeEken/Image-Generator/assets/96302110/b08a3efa-bfa6-47b5-b73a-d0d269014b51">

## UNTRAINED RESULT
<img width="239" alt="image" src="https://github.com/EgeEken/Image-Generator/assets/96302110/14a2da83-1ca0-4cd6-99c8-87d2dd54f297">

## TRAINED RESULTS (ordered by epoch count)
<img width="239" alt="image" src="https://github.com/EgeEken/Image-Generator/assets/96302110/f2dc55cb-49e4-43f6-b22f-78fcefd7da20">
<img width="239" alt="image" src="https://github.com/EgeEken/Image-Generator/assets/96302110/08788f3f-41e9-46f8-913c-01def0872b03">
<img width="238" alt="image" src="https://github.com/EgeEken/Image-Generator/assets/96302110/c814e619-9455-41b5-9280-2c5ab4f9c86e">

