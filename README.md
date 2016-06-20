# Zookeeper Brain
Pet identification with machine learning.

The application uses three models to identify the animal in the scene.

Training Spreadsheet:
https://docs.google.com/spreadsheets/d/1q53TwJEr8OQHg-kgqHj5lO3Gm0htZGX0sZfUr19gQxM/edit#gid=1668399985

Trained using Image Trainer:
https://github.com/LastZactionHero/image_trainer_api

### 1. Identify Night/Day
Determine if it is nighttime or daytime. This uses logistic regression with the average brightness of the image.

### 2. Identify Anything Happening
Most of the time no animals are present. A network could achieve a pretty low score by always predicting that the scene is empty.

This is a static scene and animals are always in the same places, so a fully-connected network works.

### 3. Identify Specific
Identify the specific animals in the scene.

## Setup
Update the image paths in ```constants.py```

## Training

```
python train_night_day.py
python train_anything_happening.py
python train_specific.py
```

## Predicting

```
python predict_total /path/to/64x64/image.jpg
```

```
> Too dark to see.
> Nothing is happening.
> It's a Strcat!
> It's a Malloc!
> It's a Cody!
> It's a Unknown!
```
