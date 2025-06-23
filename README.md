# Facial Expression Classification using Neural Networks

## Table of Contents

* [Project Overview](#project-overview)
* [Collaborators](#collaborators)
* [Motivation](#motivation)

---

## Project Overview

This repository contains all the code, documentation, and resources for a facial expression classification project using neural networks. The goal is to build, train, and evaluate a convolutional neural network (CNN) capable of recognizing basic human emotions—happy, sad, angry, surprised, fearful, disgusted, and neutral—from grayscale face images.

Key components:

* Data ingestion and preprocessing pipeline
* CNN architecture design and training scripts
* Model evaluation and visualization tools
* This project was developed as part of a university course on neural networks and deep learning.

---

## Collaborators

* Mateusz Garncarczyk
* Jose Javier Garcia Torrejon
* Dawid Pawliczek

---

## Motivation

Understanding and classifying human facial expressions is a cornerstone of affective computing. Applications range from human-computer interaction (HCI) and driver monitoring systems to mental health assessments and social robotics. By training a neural network to accurately distinguish between common emotional states, we aim to:

1. Demonstrate proficiency in implementing CNNs for image classification.
2. Gain experience with popular datasets (e.g., FER-2013) and preprocessing techniques (normalization, augmentation).
3. Explore performance trade-offs between network depth, regularization, and data augmentation.
4. Provide a modular codebase that can be extended to more nuanced or real-time expression recognition tasks.

## Installaton

After you’ve cloned the repo, you need to:

1. **Install Poetry (if you don’t already have it).**

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   Make sure `poetry` is on your PATH (e.g. `export PATH="$HOME/.local/bin:$PATH"`).

2. **Change into your project directory** (where `pyproject.toml` lives):

   ```bash
   cd facial-expression-classification
   ```

3. **Install all dependencies via Poetry**:

   ```bash
   poetry install
   ```

   This will:

   * Create (or reuse) a virtual environment under Poetry’s control.
   * Read `pyproject.toml` and `poetry.lock` and install exactly the versions you specified (e.g. NumPy, etc.).

4. **Activate the virtualenv shell**

   ```bash
   poetry env activate
   ```

   That command should return the path to virtual env.
   e.g. source /Users/some_user/Library/Caches/pypoetry/virtualenvs/facexpr-Wp4LdDR0-py3.13/bin/activate
   Run that command.

5. **Verify that your entry-point works**. For example:

   ```bash
   poetry run facexpr-demo
   ```

   You should see the "hello world" printed in console.

6. **Run data download scripts**
   Pull in FER-2013 and preprocess into `data/preprocessed/`

   ```bash
   cd ./data
   chmod +x download_data.sh
   ./download_data.sh
   ```

   This will populate your `data/` folder (so the training/inference code can find images).
