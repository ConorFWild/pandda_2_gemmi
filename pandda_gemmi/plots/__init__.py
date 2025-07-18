import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_aligned_density_projection(
        dmaps,
        models_to_process,
        characterization_set_masks,
        output_dir,
        projection='umap'
):
    pca = PCA(n_components=min(200, len(dmaps)))
    pca_embedding = pca.fit_transform(dmaps)

    if projection =='umap':
        reducer = umap.UMAP()
    else:
        reducer = TSNE(perplexity=len(dmaps)-1)
    embedding = reducer.fit_transform(pca_embedding)
    for model_number in models_to_process:
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=[sns.color_palette()[x] for x in characterization_set_masks[model_number]],
            # s=mpl.rcParams['lines.markersize']*2
        )
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f'UMAP projection of model {model_number}', fontsize=24)
        plt.savefig(output_dir / f'model_{model_number}_umap_embedding.png')
        plt.clf()
