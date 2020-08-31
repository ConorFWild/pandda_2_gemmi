from pprint import PrettyPrinter
printer = PrettyPrinter()

from joblib import Parallel

from pandda_gemmi.config import Config
from pandda_gemmi.logs import Log, XmapLogs, ModelLogs
from pandda_gemmi.types import *


def main():
    config: Config = Config.from_args()
    log: Log = Log.from_dir(config.output.out_dir)
    printer.pprint(config)

    pandda_fs_model: PanDDAFSModel = PanDDAFSModel.from_dir(config.input.data_dirs,
                                                            config.output.out_dir,
                                                            config.input,
                                                            )


    with Parallel(n_jobs=config.params.processing.process_dict_n_cpus) as parallel:

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

        datasets: Datasets = datasets.remove_dissimilar_models(reference,
                                                               config.params.filtering.max_rmsd_to_reference,
                                                               )
        print("\tAfter filters (remove dissimilar models) {} datasets".format(len(datasets.datasets)))

        datasets: Datasets = datasets.remove_dissimilar_space_groups(reference)
        print("\tAfter filters (dissimilar spacegroups) {} datasets".format(len(datasets.datasets)))

        print("Getting grid")
        grid: Grid = Grid.from_reference(reference,
                                         config.params.masks)
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
            print(ModelLogs.from_model(model))

            print("\tGetting zmaps...")
            zmaps: Zmaps = Zmaps.from_xmaps(model=model,
                                            xmaps=xmaps,
                                            )
            print("\t\tGot {} zmaps".format(len(zmaps)))

            print("\tGetting clusters from zmaps...")
            clusterings: Clusterings = Clusterings.from_Zmaps(zmaps, reference, config.params.blob_finding,
                                                              config.params.masks)
            print("\t\tGot {} initial clusters!".format(len(clusterings)))
            clusterings: Clusterings = clusterings.filter_size(grid,
                                                               config.params.blob_finding.min_blob_volume)
            clusterings: Clusterings = clusterings.filter_peak(grid,
                                                               config.params.blob_finding.min_blob_z_peak)

            events: Events = Events.from_clusters(clusterings)

            z_map_files: ZMapFiles = ZMapFiles.from_zmaps(zmaps)
            event_map_files: EventMapFiles = EventMapFiles.from_events(events,
                                                                       xmaps,
                                                                       )

        site_table_file: SiteTableFile = SiteTableFile.from_events(events)
        event_table_file: EventTableFile = EventTableFile.from_events(events)


if __name__ == "__main__":
    main()
