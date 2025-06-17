import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_aligned_density_projection(
        dmaps,
        models_to_process,
        characterization_set_masks,
        output_dir
):
    pca = PCA(n_components=200)
    pca_embedding = pca.fit_transform(dmaps)

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(pca_embedding)
    for model_number in models_to_process:
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=[sns.color_palette()[x] for x in characterization_set_masks[model_number]])
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f'UMAP projection of model {model_number}', fontsize=24);
        plt.savefig(output_dir / f'model_{model_number}_umap_embedding.png')
