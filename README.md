# Sketch to Image Generator

A custom pix2pix GAN model that generates photographs of birds from a user's input sketch, fully local training and inference.

This project was both to get a deeper understanding of image generation, and to gain MLops experience, and I think i got experience with most if not all of the full stack here:

- Dataset from **Kaggle**
- Trained with **PyTorch**
- Containerized with **Docker**
- Experiments monitored with **MLflow**
- Interface built with **Gradio**
- Demo hosted on **[HuggingFace Spaces](https://huggingface.co/spaces/EgeEken/Bird_Sketch2Image)**

https://github.com/user-attachments/assets/44773dd6-58b5-42d1-b725-3ad87ec9dea5

# Example use cases 

## Generating an image of a bird sitting on a table 
<img width="1188" height="712" alt="image" src="https://github.com/user-attachments/assets/066024bf-ee03-40c9-bb2d-67900799501e" />

## Penguin Generator Model
### Models can be trained on specific bird species to have more consistent/specific generation
Or any type of image for that matter, it's a GAN.
<img width="1167" height="554" alt="image" src="https://github.com/user-attachments/assets/02916fbb-a07a-4d7e-adcb-02c5cc73eea2" />
(This specific penguin generator model in the repository as `penguin_generator-model.pth`
