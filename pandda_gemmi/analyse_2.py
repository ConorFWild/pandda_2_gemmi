# Base python
import os
import traceback
import time
import pprint
from functools import partial
import json
import pickle

# Scientific python libraries
import joblib

import ray

## Custom Imports
from pandda_gemmi import constants
from pandda_gemmi.common import Partial
from pandda_gemmi.args import PanDDAArgs
from pandda_gemmi.dataset.dataset import GetReferenceDataset
from pandda_gemmi.edalignment.grid import GetGrid
from pandda_gemmi.pandda_logging import STDOUTManager, log_arguments, PanDDAConsole
from pandda_gemmi.dependencies import check_dependencies
from pandda_gemmi.dataset import (
    StructureFactors,
    SmoothBFactors,
    GetDatasetsStatistics,
    drop_columns,
    GetDatasets,
    GetReferenceDataset,
    PostprocessDatasets
)
from pandda_gemmi.edalignment import (GetGrid, GetAlignments,
                                      LoadXmap, LoadXmapFlat, GetXmaps
                                      )
from pandda_gemmi.filters import (
    FiltersDataQuality,
    FiltersReferenceCompatibility,
    FilterNoStructureFactors,
    FilterResolutionDatasets,
    FilterRFree,
    FilterDissimilarModels,
    FilterIncompleteModels,
    FilterDifferentSpacegroups,
    DatasetsValidator
)
from pandda_gemmi.comparators import (
    GetComparatorsHybrid, GetComparatorsHighResFirst, GetComparatorsHighResRandom, GetComparatorsHighRes,
    GetComparatorsCluster)
from pandda_gemmi.shells import get_shells_multiple_models
from pandda_gemmi.logs import (
    save_json_log,
)

from pandda_gemmi.pandda_functions import (
    get_dask_client,
    process_global_serial,
    process_global_dask,
    get_shells,
    get_common_structure_factors,
    GetStructureFactors,
)
from pandda_gemmi.event import GetEventScoreInbuilt, add_sites_to_events
from pandda_gemmi.ranking import (
    GetEventRankingAutobuild,
    GetEventRankingSize,
    GetEventRankingSizeAutobuild
)
from pandda_gemmi.autobuild import (
    merge_ligand_into_structure_from_paths,
    save_pdb_file,
    GetAutobuildResultRhofit,
    GetAutobuilds
)
from pandda_gemmi.tables import (
    GetEventTable,
    GetSiteTable,
    SaveEvents
)
from pandda_gemmi.fs import GetPanDDAFSModel, GetShellDirs
from pandda_gemmi.processing import (
    process_shell,
    process_shell_multiple_models,
    analyse_model,
    ProcessLocalRay,
    ProcessLocalSerial,
    ProcessLocalSpawn,
    SummarizeRun
)
from pandda_gemmi.density_clustering import (
    GetEDClustering)
from pandda_gemmi.smiles import GetDatasetSmiles
from pandda_gemmi import event_classification
from pandda_gemmi.sites import GetSites
from pandda_gemmi.analyse_interface import *
from pandda_gemmi.pandda import *

joblib.externals.loky.set_loky_pickler('pickle')
printer = pprint.PrettyPrinter()
console = PanDDAConsole()


def configure_comparator_func(pandda_args: PanDDAArgs,
                              load_xmap_flat_func: LoadXMapFlatInterface,
                              process_local: ProcessorInterface) -> GetComparatorsInterface:
    if pandda_args.comparison_strategy == "closest":
        # Closest datasets after clustering
        raise NotImplementedError()

    elif pandda_args.comparison_strategy == "high_res":
        # Almost Old PanDDA strategy: highest res datasets

        comparators_func = GetComparatorsHighRes(
            comparison_min_comparators=pandda_args.comparison_min_comparators,
            comparison_max_comparators=pandda_args.comparison_max_comparators,
        )

    elif pandda_args.comparison_strategy == "high_res_random":

        comparators_func = GetComparatorsHighResRandom(
            comparison_min_comparators=pandda_args.comparison_min_comparators,
            comparison_max_comparators=pandda_args.comparison_max_comparators,
        )

    elif pandda_args.comparison_strategy == "high_res_first":
        # Old pandda strategy: random datasets that are higher resolution

        comparators_func = GetComparatorsHighResFirst(
            comparison_min_comparators=pandda_args.comparison_min_comparators,
            comparison_max_comparators=pandda_args.comparison_max_comparators,
        )

    elif pandda_args.comparison_strategy == "cluster":
        comparators_func = GetComparatorsCluster(
            comparison_min_comparators=pandda_args.comparison_min_comparators,
            comparison_max_comparators=pandda_args.comparison_max_comparators,
            sample_rate=pandda_args.sample_rate,
            # TODO: add option: pandda_args.resolution_cutoff,
            resolution_cutoff=3.0,
            load_xmap_flat_func=load_xmap_flat_func,
            process_local=process_local,
            debug=pandda_args.debug,
        )

    elif pandda_args.comparison_strategy == "hybrid":
        comparators_func = GetComparatorsHybrid(
            comparison_min_comparators=pandda_args.comparison_min_comparators,
            comparison_max_comparators=pandda_args.comparison_max_comparators,
            sample_rate=pandda_args.sample_rate,
            # TODO: add option: pandda_args.resolution_cutoff,
            resolution_cutoff=3.0,
            load_xmap_flat_func=load_xmap_flat_func,
            process_local=process_local,
            debug=pandda_args.debug,
        )

    else:
        raise Exception("Unrecognised comparison strategy")

    return comparators_func


def configure_processing_funcs(pandda_args):
    ...


def configure_smooth_func(pandda_args: PanDDAArgs) -> SmoothBFactorsInterface:
    smooth_func: SmoothBFactorsInterface = SmoothBFactors()

    return smooth_func


def configure_load_xmap_func(pandda_args) -> LoadXMapInterface:
    load_xmap_func = LoadXmap()
    return load_xmap_func


def configure_load_xmap_flat_func(pandda_args) -> LoadXMapFlatInterface:
    load_xmap_flat_func = LoadXmapFlat()
    return load_xmap_flat_func


def configure_analyse_model_func(pandda_args):
    analyse_model_func = analyse_model
    return analyse_model_func


def configure_filter_data_quality_func(
        filter_keys: List[str],
        datasets_validator: DatasetsValidatorInterface,
        pandda_args: PanDDAArgs,
) -> FiltersDataQualityInterface:
    filters = {}

    if "structure_factors" in filter_keys:
        filters["structure_factors"] = FilterNoStructureFactors()

    if "resolution" in filter_keys:
        filters["resolution"] = FilterResolutionDatasets(pandda_args.low_resolution_completeness)

    if "rfree" in filter_keys:
        filters["rfree"] = FilterRFree(pandda_args.max_rfree)

    return FiltersDataQuality(filters, datasets_validator)


def configure_event_class_func(pandda_args):
    ...


def configure_filter_reference_compatability_func(
        filter_keys: List[str],
        datasets_validator: DatasetsValidatorInterface,
        pandda_args: PanDDAArgs,
) -> FiltersReferenceCompatibilityInterface:
    filters = {}

    if "dissimilar_models" in filter_keys:
        print("filter models")
        filters["dissimilar_models"] = FilterDissimilarModels(pandda_args.max_rmsd_to_reference)

    if "large_gaps" in filter_keys:
        print("filter gaps")
        filters["large_gaps"] = FilterIncompleteModels()

    if "dissimilar_spacegroups" in filter_keys:
        print("filter sg")
        filters["dissimilar_spacegroups"] = FilterDifferentSpacegroups()

    return FiltersReferenceCompatibility(filters, datasets_validator)


def configure_score_events_func(pandda_args: PanDDAArgs) -> GetEventScoreInterface:
    return GetEventScoreInbuilt()


def configure_event_ranking_func(pandda_args):
    # Rank the events to determine the order the are displated in
    # with STDOUTManager('Ranking events...', f'\tDone!'):
    if pandda_args.rank_method == "size":
        event_ranking_func = GetEventRankingSize()
    elif pandda_args.rank_method == "size_delta":
        raise NotImplementedError()
        # all_events_ranked = rank_events_size_delta()
    elif pandda_args.rank_method == "cnn":
        raise NotImplementedError()
        # all_events_ranked = rank_events_cnn()

    elif pandda_args.rank_method == "autobuild":
        if not pandda_args.autobuild:
            raise Exception("Cannot rank on autobuilds if autobuild is not set!")
        else:
            event_ranking_func: EventRankingInterface = GetEventRankingAutobuild()
    elif pandda_args.rank_method == "size-autobuild":
        if not pandda_args.autobuild:
            raise Exception("Cannot rank on autobuilds if autobuild is not set!")
        else:
            event_ranking_func: EventRankingInterface = GetEventRankingSizeAutobuild(0.4)
    else:
        raise Exception(f"Ranking method: {pandda_args.rank_method} is unknown!")

    return event_ranking_func


def configure_processor_postprocess(pandda_args):
    ...


def configure_rescore_event_func(pandda_args):
    ...


def configure_autobuild_func(pandda_args):
    ...


def configure_autobuild_processor_func(pandda_args):
    ...


def process_pandda(pandda_args: PanDDAArgs, ):
    # Process args
    pandda_log = {}

    console.start_dependancy_check()
    check_dependencies(pandda_args)

    console.start_log()
    pandda_log[constants.LOG_START] = time.time()
    initial_args = log_arguments(pandda_args, )

    pandda_log[constants.LOG_ARGUMENTS] = initial_args

    # Construct the PanDDA
    get_fs_model = PanDDAGetFSModel(
        GetPanDDAFSModel(
            pandda_args.data_dirs,
            pandda_args.out_dir,
            pandda_args.pdb_regex,
            pandda_args.mtz_regex,
            pandda_args.ligand_dir_regex,
            pandda_args.ligand_cif_regex,
            pandda_args.ligand_pdb_regex,
            pandda_args.ligand_smiles_regex,
        ),
        GetDatasetSmiles(),
        console,
        pandda_log,
    )
    load_datasets = PanDDALoadDatasets(
        GetDatasets(),
        GetDatasetsStatistics(),
        GetStructureFactors(),
        console,
        pandda_log,
    )
    filter_datasets = PanDDAFilterDatasets(
        configure_filter_data_quality_func(
            [
                "structure_factors",
                "resolution",
                "rfree"
            ],
            DatasetsValidator(pandda_args.min_characterisation_datasets),
            pandda_args
        ),
        console,
        pandda_log,
    )
    get_reference = PanDDAGetReference(
        GetReferenceDataset(),
        console,
        pandda_log,
    )
    filter_reference = PanDDAFilterReference(
        configure_filter_reference_compatability_func(
            [
                "dissimilar_models",
                "large_gaps",
                # "dissimilar_spacegroups",
            ],
            DatasetsValidator(pandda_args.min_characterisation_datasets),
            pandda_args
        ),
        console,
        pandda_log,
    )
    postprocess_datasets = PanDDAPostprocessDatasets(
        PostprocessDatasets(
            SmoothBFactors(),
            configure_processor_postprocess(pandda_args)
        ),
        console,
        pandda_log,
    )
    get_grid = PanDDAGetGrid(
        GetGrid(),
        console,
        pandda_log,
    )
    get_alignments = PanDDAGetAlignments(
        GetAlignments(),
        console,
        pandda_log,
    )
    get_shell_results = PanDDAGetShellResults(
        GetShellResults(
            PanDDAGetComparators(),
            PanDDAGetShellDatasets(),
            PanDDAProcessShell(
                get_shell_datasets,
                PanDDAGetHomogenisedDatasets(
                    TruncateDatasetReflections(),
                ),
                GetXmaps(
                    LoadXmap,
                    configure_get_xmap_processor(pandda_args)
                ),
                PanDDAGetModel(
                    configure_get_model_processor(pandda_args)
                ),
                PanDDAProcessModel(
                    PanDDAGetZmap(),
                    PanDDAGetEvents(GetEDClustering()),
                    PanDDAScoreEvents(configure_score_events_func(pandda_args)),
                    PanDDAGetModelResult(),
                )
            ),
            configure_shell_processor(pandda_args)
        ),
        console,
        pandda_log,
    )
    get_autobuilds = PanDDAGetAutobuilds(
        GetAutobuilds(
            configure_autobuild_func(pandda_args),
            configure_autobuild_processor_func(pandda_args),
        ),
        console,
        pandda_log,
    )
    rescore_events = PanDDARescoreEvents(
        configure_rescore_event_func,
        console,
        pandda_log,
    )
    summarise_run = PanDDASummariseRun(
        SummarizeRun(
            configure_event_class_func(pandda_args),
            configure_event_ranking_func(pandda_args),
            GetSites(pandda_args.max_site_distance_cutoff),
            SaveEvents(),
            GetEventTable(),
            GetSiteTable(),
        ),
        console,
        pandda_log,
    )
    pandda = PanDDA(
        get_fs_model,
        load_datasets,
        filter_datasets,
        get_reference,
        filter_reference,
        postprocess_datasets,
        get_grid,
        get_alignments,
        get_shell_results,
        get_autobuilds,
        rescore_events,
        summarise_run,
    )

    ###################################################################
    # # Run the PanDDA
    ###################################################################
    pandda()


if __name__ == '__main__':
    args = PanDDAArgs.from_command_line()
    console.summarise_arguments(args)
    process_pandda(args)
