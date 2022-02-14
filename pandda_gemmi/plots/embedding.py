from sklearn import decomposition, metrics
import umap
from bokeh.plotting import ColumnDataSource, figure, output_file, show, save


def embed_umap(reduced_array):
    distance_matrix = metrics.pairwise_distances(reduced_array)
    pca = decomposition.PCA(n_components=min(distance_matrix.shape[0], 50))
    reducer = umap.UMAP()
    transform = pca.fit_transform(distance_matrix)
    transform = reducer.fit_transform(transform)
    return transform


def bokeh_scatter_plot(embedding, labels, known_apos, plot_file):
    output_file(str(plot_file))

    apos = []
    for label in labels:
        if label in known_apos:
            apos.append("green")
        else:
            apos.append("pink")

    source = ColumnDataSource(
        data=dict(
            x=embedding[:, 0].tolist(),
            y=embedding[:, 1].tolist(),
            dtag=labels,
            apo=apos
        ))

    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("dtag", "@dtag"),
        ("apo", "@apo"),
    ]

    p = figure(plot_width=1200, plot_height=1200, tooltips=TOOLTIPS,
               title="Mouse over the dots",
               )

    p.circle('x', 'y', size=15, source=source, color="apo")

    save(p)


def save_plot_pca_umap_bokeh(dataset_connectivity_matrix, labels, known_apos, plot_file):
    embedding = embed_umap(dataset_connectivity_matrix)
    bokeh_scatter_plot(embedding, labels, known_apos, plot_file)
