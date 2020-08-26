from pandda_gemmi.config import Config
from pandda_gemmi.logs import Log
from pandda_gemmi.types import *


def main():
    config: Config = Config.from_args()
    log: Log = Log.from_dir(config.output.out_dir)
    print(config)

    pandda_fs_model: PanDDAFSModel = PanDDAFSModel.from_dir(config.input.data_dirs,
                                                            config.output.out_dir,
                                                            config.input,
                                                            )

    datasets: Datasets = Datasets.from_dir(pandda_fs_model)
    print("\tGot {} datasets".format(len(datasets.datasets)))


    datasets: Datasets = datasets.remove_invalid_structure_factor_datasets(
        config.params.diffraction_data.structure_factors)
    print("\tAfter filters (structure factors) {} datasets".format(len(datasets.datasets)))
    datasets: Datasets = datasets.remove_low_resolution_datasets(
        config.params.diffraction_data.low_resolution_completeness)
    print("\tAfter filters (low resolution) {} datasets".format(len(datasets.datasets)))
    datasets: Datasets = datasets.remove_bad_rfree(config.params.filtering.max_rfree)
    print("\tAfter filters (max rfree) {} datasets".format(len(datasets.datasets)))
    datasets: Datasets = datasets.remove_bad_wilson(config.params.filtering.max_wilson_plot_z_score)  # TODO
    print("\tAfter filters {} datasets".format(len(datasets.datasets)))
    datasets: Datasets = datasets.scale_reflections()  # TODO
    print("\tAfter filters (scale reflections) {} datasets".format(len(datasets.datasets)))

    reference: Reference = Reference.from_datasets(datasets)
    print(reference)

    datasets: Datasets = datasets.remove_dissimilar_models(reference,
                                                           config.params.filtering.max_rmsd_to_reference,
                                                           )
    print("\tAfter filters (remove dissimilar models) {} datasets".format(len(datasets.datasets)))


    datasets: Datasets = datasets.remove_dissimilar_space_groups(reference)
    print("\tAfter filters (dissimilar spacegroups) {} datasets".format(len(datasets.datasets)))

    print("Getting grid")
    grid: Grid = Grid.from_reference(reference)
    print("\tGot grid")

    print("Getting alignments")
    alignments: Alignments = Alignments.from_datasets(reference,
                                                      datasets,
                                                      )
    print("\tGot alignments")

    # Shells
    print("Getting shells...")
    for shell in Shells.from_datasets(datasets, config.params.resolution_binning):
        print("\tWorking on shell {}".format(shell.res_min))

        print("\tTruncating datasets...")
        truncated_datasets: Datasets = datasets.truncate(resolution=shell.res_min)

        print("\tGetting reference map...")
        reference_map: ReferenceMap = ReferenceMap.from_reference(reference,
                                                                  alignments[reference.dtag],
                                                                  grid,
                                                                  config.params.diffraction_data.structure_factors,
                                                                  )

        print("\tGetting maps...")
        xmaps: Xmaps = Xmaps.from_aligned_datasets(truncated_datasets,
                                                   alignments,
                                                   grid,
                                                   config.params.diffraction_data.structure_factors,
                                                   )
        print("\t\tGot {} xmaps".format(len(xmaps)))

        print("\tDetermining model...")
        model: Model = Model.from_xmaps(xmaps)
        print("\t\tGot model")

        print("\tGetting zmaps...")
        zmaps: Zmaps = Zmaps.from_xmaps(model=model,
                                        xmaps=xmaps,
                                        )
        print("\t\tGot {} zmaps".format(len(zmaps)))

        print("\tGetting clusters from zmaps...")
        clusters: Clusters = Clusters.from_Zmaps(zmaps)
        print("\t\tGot {} initial clusters!".format(len(clusters)))
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
