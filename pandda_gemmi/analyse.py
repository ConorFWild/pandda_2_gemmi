from pandda_gemmi.config import Config
from pandda_gemmi.logging import Log
from pandda_gemmi.types import *


def main():
    config: Config = Config.from_args()
    log: Log = Log.from_dir(config.output.out_dir)

    pandda_fs_model: PanDDAFSModel = PanDDAFSModel.from_dir(config.input.data_dirs,
                                                            config.output.out_dir,
                                                            )

    datasets: Datasets = Datasets.from_dir(pandda_fs_model)
    reference: Reference = Reference.from_datasets(datasets)

    datasets: Datasets = datasets.remove_invalid_structure_factor_datasets(
        config.params.diffraction_data.structure_factors)
    datasets: Datasets = datasets.remove_low_resolution_datasets(
        config.params.diffraction_data.low_resolution_completeness)
    datasets: Datasets = datasets.scale_reflections()  # TODO
    datasets: Datasets = datasets.remove_dissimilar_models(reference,
                                                           config.params.filtering.max_rmsd_to_reference,
                                                           )
    datasets: Datasets = datasets.remove_bad_rfree(config.params.filtering.max_rfree)
    datasets: Datasets = datasets.remove_dissimilar_space_groups(reference)
    datasets: Datasets = datasets.remove_bad_wilson(config.params.filtering.max_wilson_plot_z_score)  # TODO

    grid: Grid = Grid.from_reference(reference)

    alignments: Alignments = Alignments.from_datasets(reference,
                                                      datasets)

    # Shells
    for shell in Shells.from_datasets(datasets, config.params.resolution_binning):
        truncated_datasets: Datasets = datasets.truncate(resolution=shell.resolution)
        reference_map: ReferenceMap = ReferenceMap.from_reference(reference)
        xmaps: Xmaps = Xmaps.from_aligned_datasets(truncated_datasets,
                                                   alignments,
                                                   grid,
                                                   config.params.diffraction_data.structure_factors,
                                                   )
        model: Model = Model.from_xmaps(xmaps)
        zmaps: Zmaps = Zmaps.from_xmaps(model=model,
                                        xmaps=xmaps,
                                        )

        clusters: Clusters = Clusters.from_Zmaps(zmaps)
        clusters: Clusters = clusters.filter_size_and_peak()
        clusters: Clusters = clusters.filter_distance_from_protein()
        clusters: Clusters = clusters.group_close()
        clusters: Clusters = clusters.remove_symetry_pairs()

        events: Events = Events.from_clusters(clusters)

        z_map_files: ZMapFiles = ZMapFiles.from_zmaps(zmaps)
        event_map_files: EventMapFiles = EventMapFiles.from_events(events,
                                                                   xmaps,
                                                                   )

    site_table_file: SiteTableFile = SiteTableFile.from_events(events)
    event_table_file: EventTableFile = EventTableFile.from_events(events)


if __name__ == "__main__":
    main()
