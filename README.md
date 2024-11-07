# Deep Autoencoder for Denoising Noisy Images

This repository contains the study of the structure of autoencoders with the aim of cleaning and reconstructing images from noisy inputs. In the project, trained on the MNIST and FMNIST datasets, different autoencoder models are tried. The model achieves impressive denoising capabilities, showcasing the potential of deep learning in noise reduction.


## Project Structure
- [`denoising_autoencoder_FMNIST.ipynb`](./denoising_autoencoder_FMNIST.ipynb): Jupyter Notebook with the process of achieving the best autoencoder structure and training for the **FMNIST** dataset.
- [`denoising_autoencoder_MNIST.ipynb`](./denoising_autoencoder_MNIST.ipynb): Jupyter Notebook with the process of achieving the best autoencoder structure and training for the **MNIST** dataset.


## Requirements
- Python 3.7 or higher
- Jupyter Notebook


## Methods and Techniques
- **Data Preprocessing**: Text cleaning, tokenization and lemmatization.
- **Feature Extraction**: Text Vectorization Techniques by implementing TFIDF, Word2Vec, BERT embeddings.
- **Classification Models**: Random Forest and BERT for document classifcation.

## Dataset
The MNIST and FMNIST datasets were used to evaluate the model's effectiveness. Noise was added to these datasets to create training data, with noise variances ranging from 0.1 to 2, enabling the model to learn to recover clean images.

## Model Architecture
- **Encoder-Decoder Structure**: The autoencoder features a 3-layer encoder and mirrored decoder structure with ReLU activations, aimed at reducing dimensionality and reconstructing clean images.
- **Loss Function**: Mean Squared Error (MSE) with Lasso regularization.
- **Training Configuration**: The autoencoder was trained with varying layer depths and projected dimensions to determine the optimal architecture.

## Results
The autoencoder effectively reduced noise, as demonstrated by PSNR and visual comparisons. For optimal performance, a 3-layer architecture with a 50-dimensional bottleneck was used.

### Usage
For looking at the study of the different models tried, open one of the Jupyter Notebooks to view and run the code:

- jupyter notebook denoising_autoencoder_FMNIST.ipynb
- jupyter notebook denoising_autoencoder_MNIST.ipynb

In the own notebook there is cell for the import of the neccessary libraries.

You can also try the trained model by just running the cell:
```python
best_model_path = f'denoising_autoencoder_fmnist.pth'

autoencoder_denoising.load_state_dict(torch.load(best_model_path))
```
```python
best_model_path = f'denoising_autoencoder_mnist.pth'

autoencoder_denoising.load_state_dict(torch.load(best_model_path))
```
And then running the last part of the notebook.

You can also try one of the other models tried already trained, by loading them the same way. They can be found at the models folder.

## Acknowledgments
This project was developed as part of the Master's program in Telecommunications Engineering at Universidad Carlos III de Madrid (UC3M). Special thanks to my collaborators [Elena Almagro](https://linkedin.com/in/elenaalmagro/) and [Juan Muñoz](https://www.linkedin.com/in/juan-munoz-villalon/) for their valuable contributions and teamwork throughout the project.
