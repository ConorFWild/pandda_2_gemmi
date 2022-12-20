###################################################################
# # Run Info
###################################################################
VERSION = "0.1.0"

###################################################################
# # Pandda File names
###################################################################

PANDDA_ANALYSES_DIR = "analyses"
PANDDA_SHELL_DIR = "shells"
PANDDA_ANALYSE_EVENTS_FILE = "pandda_analyse_events.csv"
PANDDA_ANALYSE_SITES_FILE = "pandda_analyse_sites.csv"
PANDDA_HTML_SUMMARIES_DIR = "html_summaries"
PANDDA_PROCESSED_DATASETS_DIR = "processed_datasets"
PANDDA_MODELLED_STRUCTURES_DIR = "modelled_structures"
PANDDA_LIGAND_FILES_DIR = "ligand_files"
PANDDA_PDB_FILE = "{}-pandda-input.pdb"
PANDDA_MTZ_FILE = "{}-pandda-input.mtz"
PANDDA_TEXT_LOG_FILE = "pandda_log.txt"
PANDDA_EVENT_JSON_FILE = "events.json"

PANDDA_LIGAND_CIF_FILE = "ligand.cif"
PANDDA_LIGAND_PDB_FILE = "ligand.pdb"
PANDDA_LIGAND_SMILES_FILE = "ligand.smiles"

PANDDA_INSPECT_EVENTS_PATH = "pandda_inspect_events.csv"
PANDDA_Z_MAP_FILE = "{dtag}-z_map.native.ccp4"
PANDDA_EVENT_MAP_FILE = "{dtag}-event_{event_idx}_1-BDC_{bdc}_map.native.ccp4"
PANDDA_EVENT_MODEL = "{}-pandda-model.pdb"

PANDDA_PROTEIN_MASK_FILE = "protein_mask.ccp4"
PANDDA_SYMMETRY_MASK_FILE = "symmetry_mask.ccp4"
PANDDA_TOTAL_MASK_FILE = "total_mask.ccp4"
PANDDA_MEAN_MAP_FILE = "mean_{number}_{res}.ccp4"
PANDDA_SIGMA_S_M_FILE = "sigma_s_m_{number}_{res}.ccp4"

###################################################################
# # Logging constants
###################################################################
COMMON_F_PHI_LABEL_PAIRS = (
    ("FWT", "PHWT"),
    ("2FOFCWT", "PH2FOFCWT"),
    ("2FOFCWT_iso-fill", "PH2FOFCWT_iso-fill"),
    ("2FOFCWT_fill", "PH2FOFCWT_fill",),
    ("2FOFCWT", "PHI2FOFCWT")
)

###################################################################
# # Args
###################################################################

ARGS_DESCRIPTION = (
    "Perform a Pan-Dataset Density Analysis.\n"
    "\n"
    "Typical usage is:\n"
    "\n"
    "python /path/to/analyse.py <data dirs> <output dirs> --pdb_regex='dimple.pdb' --mtz_regex='dimple.mtz' "
    "--autobuild=True <options>\n"
)
ARGS_DATA_DIRS = "--data_dirs"
ARGS_DATA_DIRS_HELP = "The path to the directory containing the dataset directories. This directory must only contain other directories, one for each crystal to be analysed in PanDDA, each with a unique name and containing a file for the reflections from that crystal and a model of those reflections. The name of that model must match the pattern given in --pdb_regex and the name of the reflections file must match the pattern given in --mtz_regex"
ARGS_OUT_DIR = "--out_dir"
ARGS_OUT_DIR_HELP = "The path to the directory in which to store the results of the PanDDA."

ARGS_PDB_REGEX = "--pdb_regex"
ARGS_PDB_REGEX_HELP = "A grep pattern matching the name of the model of the crystal data in each crystal directory in the directory given by --data_dirs."
ARGS_PDB_REGEX_DEFAULT = "dimple.pdb"

ARGS_MTZ_REGEX = "--mtz_regex"
ARGS_MTZ_REGEX_HELP = "A grep pattern matching the name of the reflections of each crystal in each crystal directory " \
                      "in the directory given by --data_dirs. "
ARGS_MTZ_REGEX_DEFAULT = "dimple.mtz"

ARGS_LIGAND_CIF_REGEX = "--ligand_cif_regex"
ARGS_LIGAND_CIF_REGEX_HELP = "A grep pattern matching the restraints associated with each ligand that might be bound " \
                             "in the crystal."
ARGS_LIGAND_CIF_REGEX_DEFAULT = "[0-9a-zA-Z-]+[.]cif"

ARGS_LIGAND_SMILES_REGEX = "--ligand_smiles_regex"
ARGS_LIGAND_SMILES_REGEX_HELP = "A grep pattern matching the smiles associated with each ligand that may be bound in " \
                                "the crystal."
ARGS_LIGAND_SMILES_REGEX_DEFAULT = "[0-9a-zA-Z-]+[.]smiles"

ARGS_DATA_QUALITY_FILTERS = "--data_quality_filters"
ARGS_DATA_QUALITY_FILTERS_HELP = "A comma seperated list of filters to apply. May include \"structure_factors\", " \
                                 "\"resolution\" and \"rfree\"."
ARGS_DATA_QUALITY_FILTERS_DEFAULT = ("structure_factors", "resolution", "rfree")

ARGS_REFERENCE_COMPARABILITY_FILTERS = "--reference_comparability_filters"
ARGS_REFERENCE_COMPARABILITY_FILTERS_HELP = "A comma seperated list of filters to apply. May include " \
                                            "\"dissimilar_models\", \"large_gaps\" and \"dissimilar_spacegroups\""
ARGS_REFERENCE_COMPARABILITY_FILTERS_DEFAULT = ("dissimilar_models", "large_gaps", )

ARGS_LIGAND_PDB_REGEX = "--ligand_pdb_regex"
ARGS_LIGAND_PDB_REGEX_HELP = "A grep pattern matching the pdb associated with each ligand that may be " \
                             "bound in the crystal."
ARGS_LIGAND_PDB_REGEX_DEFAULT = "[0-9a-zA-Z-]+[.]pdb"

ARGS_LIGAND_DIR_REGEX = "--ligand_dir_regex"
ARGS_LIGAND_DIR_REGEX_HELP = "A grep pattern matching a directory in each crystal directory in the directory given by --data_dirs. If this is given, then other cif regexs will only be searched for inside this directory, if it can be found."
ARGS_LIGAND_DIR_REGEX_DEFAULT = "compound"
ARGS_LOCAL_PROCESSING = "--local_processing"
ARGS_LOCAL_PROCESSING_HELP = "A string from 'serial', 'joblib', 'multiprocessing_forkserver' or " \
                             "'multiprocessing_spawn' that gives how node-local parallelism should be used in the " \
                             "program. If serial, then no parallelism will be used. If joblib, then a pool of joblib " \
                             "workers will handle multiprocessing. If multiprocessing_forkserver, then a forkserver " \
                             "will handle multiprocessing. If multiprocessing_spawn then spaened processes will be " \
                             "used."
ARGS_LOCAL_CPUS = "--local_cpus"
ARGS_LOCAL_CPUS_HELP = "An integer that gives number of node-local cpus to use for multiprocessing."
ARGS_GLOBAL_PROCESSING = "--global_processing"
ARGS_GLOBAL_PROCESSING_HELP = "A string from 'serial' and 'distributed'. that gives how to handle processing each " \
                              "resolution shell. If serial then shells will be processed on one computer. If " \
                              "distributed then a dask scheduler will attempt to exploit cluster resources to process shells simulatainiously."
ARGS_GLOBAL_PROCESSING_DEFAULT = "serial"
ARGS_MEMORY_AVAILABILITY = "--memory_availability"
ARGS_MEMORY_AVAILABILITY_HELP = "A string from 'very low', 'low' or 'high' that gives the tradoff between performance and RAM usage in processing resolution shells. If 'very_low' then no multiprocessing will be exploited. If 'low' then datasets will be processed in serial but parallelism will be used within processing a dataset. If high then datasets will be processed in parallel."
ARGS_JOB_PARAMS_FILE = "--job_params_file"
ARGS_JOB_PARAMS_FILE_HELP = "DEPRECATED."
ARGS_LOW_MEMORY = "--low_memory"
ARGS_LOW_MEMORY_HELP = "DEPREACTED. "
ARGS_DISTRIBUTED_SCHEDULER = "--distributed_scheduler"
ARGS_DISTRIBUTED_SCHEDULER_HELP = "A string from 'HTCONDOR', 'PBS', 'SGE', 'SLURM' that gives the name of which " \
                                  "cluster to " \
                                  "attempt to use for distributed computing of resolution shells."
ARGS_DISTRIBUTED_QUEUE = "--distributed_queue"
ARGS_DISTRIBUTED_QUEUE_HELP = "A string that gives the name of the SGE queue to submit shells to."
ARGS_DISTRIBUTED_PROJECT = "--distributed_project"
ARGS_DISTRIBUTED_PROJECT_HELP = "A string that gives the name of the SGE project to submit jobs under."
ARGS_DISTRIBUTED_NUM_WORKERS = "--distributed_num_workers"
ARGS_DISTRIBUTED_NUM_WORKERS_HELP = "An integer that gives the number of nodes to attempt to use for distributed " \
                                    "computing."
ARGS_DISTRIBUTED_CORES_PER_WORKER = "--distributed_cores_per_worker"
ARGS_DISTRIBUTED_CORES_PER_WORKER_HELP = "An integer that gives the number of cores per node for distributed computing."
ARGS_DISTRIBUTED_MEM_PER_CORE = "--distributed_mem_per_core"
ARGS_DISTRIBUTED_MEM_PER_CORE_HELP = "An integer that gives the number of GB of RAM to request for each " \
                                     "distributed process."
ARGS_DISTRIBUTED_MEM_PER_CORE_DEFAULT = 15

ARGS_DISTRIBUTED_RESOURCE_SPEC = "--distributed_resource_spec"
ARGS_DISTRIBUTED_RESOURCE_SPEC_HELP = "A string that gives which resources to request for a distributed SGE job."
ARGS_DISTRIBUTED_TMP = "--distributed_tmp"
ARGS_DISTRIBUTED_TMP_HELP = "A path to to a directory to store temporary data for distributed computing."
ARGS_JOB_EXTRA = "--job_extra"
ARGS_JOB_EXTRA_HELP = "A string that gives any extra options to give to the dask scheduler for distributed computing."
ARGS_DISTRIBUTED_WALLTIME = "--distributed_walltime"
ARGS_DISTRIBUTED_WALLTIME_HELP = "A string that gives the walltime limit for distributed computing."
ARGS_DISTRIBUTED_WATCHER = "--distributed_watcher"
ARGS_DISTRIBUTED_WATCHER_HELP = "A boolean flag that gives whether or not to assign a watcher node to each process."
ARGS_DISTRIBUTED_SLURM_PARTITION = "--distributed_slurm_partition"
ARGS_DISTRIBUTED_SLURM_PARTITION_HELP = "A string that gives which slurm partition to submit jobs under."
ARGS_GROUND_STATE_DATASETS = "--ground_state_datasets"
ARGS_GROUND_STATE_DATASETS_HELP = "A comma seperated list of dtags to use for characterising the ground state."
ARGS_EXCLUDE_FROM_Z_MAP_ANALYSIS = "--exclude_from_z_map_analysis"
ARGS_EXCLUDE_FROM_Z_MAP_ANALYSIS_HELP = "A comma seperated list of dtags to not search for events in."
ARGS_EXCLUDE_FROM_CHARACTERISATION = "--exclude_from_characterisation"
ARGS_EXCLUDE_FROM_CHARACTERISATION_HELP = "A comma seperated list of dtags to not include in ground state " \
                                          "characterisation."
ARGS_ONLY_DATASETS = "--only_datasets"
ARGS_ONLY_DATASETS_HELP = "A comma seperated list of dtags to anaylse. All other dtags will be ignored."
ARGS_IGNORE_DATASETS = "--ignore_datasets"
ARGS_IGNORE_DATASETS_HELP = "A comma seperated list of dtags to not include in amalysis."
ARGS_MAX_RMSD_TO_REFERENCE = "--max_rmsd_to_reference"
ARGS_MAX_RMSD_TO_REFERENCE_HELP = "A float giving the RMSD to the selected reference model at which beyond which to " \
                                  "exclude datasets from analysis."
ARGS_MAX_RFREE = "--max_rfree"
ARGS_MAX_RFREE_HELP = "A float giving the maximum rfree at which to exclude a dataset from analysis."
ARGS_MAX_WILSON_PLOT_Z_SCORE = "--max_wilson_plot_z_score"
ARGS_MAX_WILSON_PLOT_Z_SCORE_HELP = "NOT YET IMPLEMENTED."
ARGS_SAME_SPACE_GROUP_ONLY = "--same_space_group_only"
ARGS_SAME_SPACE_GROUP_ONLY_HELP = "A bool giving whether or not to exclude datasets from analysis if they have a " \
                                  "different spacegroup from the reference model."
ARGS_SIMILAR_MODELS_ONLY = "--similar_models_only"
ARGS_SIMILAR_MODELS_ONLY_HELP = "A bool giving whether or not to exclude datasets whose model is not sufficiently " \
                                "similar."
ARGS_COMPARISON_STRATEGY = "--comparison_strategy"
ARGS_COMPARISON_STRATEGY_HELP = "A string from 'closest', 'closest_cutoff', 'closest_cluster', 'high_res', " \
                                "'high_res_random', 'get_comparators_closest_apo_cutoff', 'local'. If 'closest', " \
                                "then each dataset will be compared to the closest datasets by RMSD. If " \
                                "'closest_cutoff the the datasets will be compared to those datasets that are closest below a certain resolution cutoff'. If 'closest_cluster' then the datasets will be compared against the closest cluster of datasets. If 'high_res' then the datasets will be compared against the highest reoslution datasets. If 'high_res_random' then each dataset will be compared against a random selection of datasets of a higher resolution that it. if 'get_comparators_closest_apo_cutoff' then each dataset will be characterised against the closest apos under a cutoff. 'local' is deprecated."
ARGS_COMPARISON_RES_CUTOFF = "--comparison_res_cutoff"
ARGS_COMPARISON_RES_CUTOFF_HELP = "A boolean value that gives the maximum resolution above that of the current " \
                                  "dataset to allow when --comparison_strategy is one of 'cluster_cutoff' or " \
                                  "'get_comparators_closest_apo_cutoff'."
ARGS_COMPARISON_MIN_COMPARATORS = "--comparison_min_comparators"
ARGS_COMPARISON_MIN_COMPARATORS_HELP = "An integer that gives The minimum number of comparators for each dataset."
ARGS_COMPARISON_MAX_COMPARATORS = "--comparison_max_comparators"
ARGS_COMPARISON_MAX_COMPARATORS_HELP = "An integer that gives the maximum number of comparators for each dataset."
ARGS_KNOWN_APOS = "--known_apos"
ARGS_KNOWN_APOS_HELP = "A list of dtags which give those datasets known not to have a fragment bound."
ARGS_EXCLUDE_LOCAL = "--exclude_local"
ARGS_EXCLUDE_LOCAL_HELP = "An integer that gives the number of closest datasets to exclude when --comparison_strategy " \
                          "is one of 'closest' or 'closest_cutoff'"
ARGS_CLUSTER_SELECTION = "--cluster_selection"
ARGS_CLUSTER_SELECTION_HELP = "A string from 'close', 'center', 'far', 'next' or 'narrow' which gives the strategy " \
                              "to use when --comparison_strategy is one of 'closest' or 'closest_cutoff'. If 'close' " \
                              "then the closest cluster will be used. If 'center' then the centermost cluster will be used. If 'far' then the furthest cluster will be used. If 'next' then the second closest cluster will be used. If 'narrow' then the most consistent cluster will be used."
ARGS_DYNAMIC_RES_LIMITS = "--dynamic_res_limits"
ARGS_DYNAMIC_RES_LIMITS_HELP = "A boolean that gives whether or not to set the resolution ranges for shells dynamicaly."
ARGS_HIGH_RES_UPPER_LIMIT = "--high_res_upper_limit"
ARGS_HIGH_RES_UPPER_LIMIT_HELP = "A float that gives the highest resolution datasets to conside for processing."
ARGS_HIGH_RES_LOWER_LIMIT = "--high_res_lower_limit"
ARGS_HIGH_RES_LOWER_LIMIT_HELP = "A float that gives the lowest resolution datasets to consider processing"
ARGS_HIGH_RES_INCREMENT = "--high_res_increment"
ARGS_HIGH_RES_INCREMENT_HELP = "A float which gives the maximum width of a resolution shell to process."
ARGS_MAX_SHELL_DATASETS = "--max_shell_datasets"
ARGS_MAX_SHELL_DATASETS_HELP = "An integer which gives that maximum number of datasets to process in one shell."
ARGS_MIN_CHARACTERISATION_DATASETS = "--min_characterisation_datasets"
ARGS_MIN_CHARACTERISATION_DATASETS_HELP = "An integer which gives the minimum number of datasets to consider " \
                                          "processing in a shell."
ARGS_STRUCTURE_FACTORS = "--structure_factors"
ARGS_STRUCTURE_FACTORS_HELP = "A string which gives the structure factors to use. It should be of the form 'FWT,PHWT'."
ARGS_ALL_DATA_ARE_VALID_VALUES = "--all_data_are_valid_values"
ARGS_ALL_DATA_ARE_VALID_VALUES_HELP = "A boolean value that gives whether or not to fail when some data are invalid.."
ARGS_LOW_RESOLUTION_COMPLETENESS = "--low_resolution_completeness"
ARGS_LOW_RESOLUTION_COMPLETENESS_HELP = "A float which gives the minimum resolution for the dataset to be complete to."
ARGS_SAMPLE_RATE = "--sample_rate"
ARGS_SAMPLE_RATE_HELP = "A float which gives the amount of upsampling to perform in FFTs."
ARGS_STATMAPS = "--statmaps"
ARGS_STATMAPS_HELP = "A boolean value which gives whether or not to output mean maps."
ARGS_RESOLUTION_FACTOR = "--resolution_factor"
ARGS_RESOLUTION_FACTOR_HELP = "DEPRECATED"
ARGS_GRID_SPACING = "--grid_spacing"
ARGS_GRID_SPACING_HELP = "DEPRECATED"
ARGS_PADDING = "--padding"
ARGS_PADDING_HELP = "DEPRECATED"
ARGS_DENSITY_SCALING = "--density_scaling"
ARGS_DENSITY_SCALING_HELP = "DEPRECATED"
ARGS_OUTER_MASK = "--outer_mask"
ARGS_OUTER_MASK_HELP = "The radius around the protein to consider for analysis."
ARGS_INNER_MASK = "--inner_mask"
ARGS_INNER_MASK_HELP = "The radius around the protein to consider as protein density."
ARGS_INNER_MASK_SYMMETRY = "--inner_mask_symmetry"
ARGS_INNER_MASK_SYMMETRY_HELP = "The radius around symmetry copies of the protein to exclude from analysis."
ARGS_CONTOUR_LEVEL = "--contour_level"
ARGS_CONTOUR_LEVEL_HELP = "A float giving the contour level at which to truncate z maps when looking for events."
ARGS_NEGATIVE_VALUES = "--negative_values"
ARGS_NEGATIVE_VALUES_HELP = "A boolean giving whether or not to cluster unusually low density."
ARGS_MIN_BLOB_VOLUME = "--min_blob_volume"
ARGS_MIN_BLOB_VOLUME_HELP = "A float giving the minimum volume of a blob to be considered an event."
ARGS_MIN_BLOB_Z_PEAK = "--min_blob_z_peak"
ARGS_MIN_BLOB_Z_PEAK_HELP = "A float giving the minimum z value in a blob to be considered an event."
ARGS_CLUSTERING_CUTOFF = "--clustering_cutoff"
ARGS_CLUSTERING_CUTOFF_HELP = "DEPRECATED."
ARGS_CLUSTER_CUTOFF_DISTANCE_MULTIPLIER = "--cluster_cutoff_distance_multiplier"
ARGS_CLUSTER_CUTOFF_DISTANCE_MULTIPLIER_HELP = "A float which gives what multiple of a diagonal lattice step a high z value can be from a blob while still being considered for inclusion."
ARGS_EVENT_SCORE = "--event_score"
ARGS_EVENT_SCORE_HELP = "Method to score events: either inbuilt or size"
ARGS_EVENT_SCORE_DEFAULT = "inbuilt"
ARGS_MAX_SITE_DISTANCE_CUTOFF = "--max_site_distance_cutoff"
ARGS_MAX_SITE_DISTANCE_CUTOFF_HELP = "The maximum distance between events for them to be considered for inclusion in " \
                                     "the same site"
ARGS_MIN_BDC = "--min_bdc"
ARGS_MIN_BDC_HELP = "A float giving the minimum background correction to be considered"
ARGS_MAX_BDC = "--max_bdc"
ARGS_MAX_BDC_HELP = "A float giving the the maximum background correction to be considered."
ARGS_INCREMENT = "--increment"
ARGS_INCREMENT_HELP = "A float giving the increment in bdcs to be considered between the min and max."
ARGS_OUTPUT_MULTIPLIER = "--output_multiplier"
ARGS_OUTPUT_MULTIPLIER_HELP = "DEPRECATED"
ARGS_AUTOBUILD = "--autobuild"
ARGS_AUTOBUILD_HELP = "A boolean giving whether or not to attempt to autobuild events."
ARGS_AUTOBUILD_STRATEGY = "--autobuild_strategy"
ARGS_AUTOBUILD_STRATEGY_HELP = "A string giving what autobuild strategy to attempt. Currently only 'rhofit' is " \
                               "implemented."
ARGS_AUTOBUILD_STRATEGY_DEFAULT = "rhofit"
ARGS_RHOFIT_COORD = "--rhofit_coord"
ARGS_RHOFIT_COORD_HELP = "A boolean value giving whether to run rhofit with a seed coord (True) or with a map with " \
                         "the density around the event masked." \
                         "density around the event (False)."
ARGS_CIF_STRATEGY = "--cif_strategy"
ARGS_CIF_STRATEGY_HELP = "A string giving the strategy with which to generate the ligand cif for autobuilding from " \
                         "'default', 'elbow', 'grade' and 'grade2'. If default then the cifs found in the dataset " \
                         "directories will be used, otherwise they will be generated from the smiles found." \
                         "otherwise the smiles "
ARGS_CIF_STRATEGY_DEFAULT = "default"
ARGS_RANK_METHOD = "--rank_method"
ARGS_RANK_METHOD_HELP = "A string giving the ranking method to be used from 'size' and 'autobuild'. If 'size' then " \
                        "the size of event will be used. If 'autobuild' then the scores from autobuilding will be used."
ARGS_DEBUG = "--debug"
ARGS_DEBUG_HELP = "A boolean value giving whether or not to print debugging information."

##########################
ARGS_MIN_CHARACTERISATION_DATASETS_DEFAULT: int = 25
ARGS_COMPARISON_STRATEGY_DEFAULT: str = "hybrid"
ARGS_LOCAL_PROCESSING_DEFAULT: str = "multiprocessing_spawn"
ARGS_LOCAL_CPUS_DEFAULT: int = 6
ARGS_MEMORY_AVAILABILITY_DEFAULT: str = "high"
ARGS_AUTOBUILD_DEFAULT: bool = True
ARGS_RANK_METHOD_DEFAULT: str = "event_score"
ARGS_MIN_BDC_DEFAULT: float = 0.0
ARGS_MAX_BDC_DEFAULT: float = 0.95
ARGS_MAX_SITE_DISTANCE_CUTOFF_DEFAULT: float = 1.732

###################################################################
# # Console constants
###################################################################
CONSOLE_START_DEPENDENCY_CHECK = "Checking dependencies..."
CONSOLE_START_LOG = "Initialising log..."
CONSOLE_START_INIT_SHELL_PROCCESS = "Getting shell processor..."
CONSOLE_START_INIT_MULTIPROCESS = "Getting multiprocessor..."
CONSOLE_START_FS_MODEL = "Building model of file system..."
CONSOLE_START_LOAD_DATASETS = "Loading datasets..."
CONSOLE_START_QUALITY_FILTERS = "Filtering datasets based on data quality..."
CONSOLE_START_REF_SELEC = "Selecting reference structure for alignment..."
CONSOLE_START_B_FACTOR_SMOOTHING = "Performing B factor smoothing of reflections..."
CONSOLE_START_REF_COMPAT_FILTERS = "Filtering datasets on compatibility with reference..."
CONSOLE_START_GET_GRID = "Getting grid to perform analysis on..."
CONSOLE_START_ALIGN = "Aligning datasets to reference..."
CONSOLE_START_GET_COMPAR = "Getting comparators for each dataset..."
CONSOLE_START_PROCESS_SHELLS = "Processing resolution shells..."
CONSOLE_START_AUTOBUILDING = "Autobuilding event maps..."
CONSOLE_START_RANKING = "Ranking events..."
CONSOLE_START_ASSIGN_SITES = "Assigning sites..."
CONSOLE_START_SUMMARY = "Summarising PanDDA run..."
CONSOLE_START_EVENT_CLASSIFICATION = "Starting classification of PanDDA Events..."

###################################################################
# # Logging constants
###################################################################
PANDDA_LOG_FILE = "pandda_log.json"

LOG_ARGUMENTS: str = "The arguments to the main function and their values"
LOG_START: str = "Start time"
LOG_TRACE: str = "trace"
LOG_EXCEPTION: str = "exception"
LOG_TIME: str = "Time taken to complete PanDDA"

LOG_INVALID: str = "Datasets filtered for being invalid"
LOG_LOW_RES: str = "Datasets filtered for being too low res"
LOG_RFREE: str = "Datasets filtered for having high RFree"
LOG_DISSIMILAR_STRUCTURE: str = "Datasets filtered for having dissimilar structures"
LOG_GAPS: str = "Datasets filtered for having large gaps"
LOG_SG: str = "Datasets filtered for having a different spacegroup"

LOG_DATASETS: str = "Summary of input datasets"

LOG_KNOWN_APOS: str = "Known apo dtags"

LOG_SHELL_XMAP_TIME: str = "Time taken to generate aligned xmaps"
LOG_SHELL_DATASET_LOGS: str = "Logs for each dataset in shell"
LOG_SHELL_TIME: str = "Time taken to process shell"
LOG_SHELLS: str = "Logs for each shell"
LOG_SHELL_DATASETS: str = "Datasets in shell"

LOG_DATASET_TIME: str = "TIme taken to process dataset"
LOG_DATASET_TRAIN: str = "Datasets density charactersied against"
LOG_DATASET_MEAN: str = "Mean map statistics"
LOG_DATASET_SIGMA_I: str = "Sigma I's for dataset"
LOG_DATASET_SIGMA_S: str = "Sigma S map statistics"
LOG_DATASET_MODEL_TIME: str = "Time taken to generate dataset model"
LOG_DATASET_Z_MAPS_TIME: str = "Time taken to generate z map"
LOG_DATASET_INITIAL_CLUSTERS_NUM: str = "Initial number of clusters"
LOG_DATASET_LARGE_CLUSTERS_NUM: str = "Number of clusters after filtering small clusters"
LOG_DATASET_PEAKED_CLUSTERS_NUM: str = "Number of clusters after filtering low peaked clusters"
LOG_DATASET_MERGED_CLUSTERS_NUM: str = "Number of clusters after merging clusters"
LOG_DATASET_CLUSTER_TIME: str = "Time taken to cluster z map"
LOG_DATASET_EVENT_TIME: str = "Time taken to get events"
LOG_DATASET_EVENT_MAP_TIME: str = "Time taken to generate event map"
LOG_DATASET_XMAP_TIME: str = "Time taken to generate aligned xmaps"
LOG_DATASET_CLUSTER_SIZES: str = "Size of initial clusters"

LOG_AUTOBUILD_TIME: str = "Time taken to autobuild events"
LOG_AUTOBUILD_SELECTED_BUILDS: str = "Build selected for each dataset"
LOG_AUTOBUILD_SELECTED_BUILD_SCORES: str = "Score for Build selected for each dataset"
LOG_AUTOBUILD_COMMANDS: str = "Commands to autobuild each event"

###################################################################
# # Residue names
###################################################################

RESIDUE_NAMES = ["ALA",
                 "ARG",
                 "ASN",
                 "ASP",
                 "CYS",
                 "GLN",
                 "GLU",
                 "HIS",
                 "ILE",
                 "LEU",
                 "LYS",
                 "MET",
                 "PHE",
                 "PRO",
                 "SER",
                 "THR",
                 "TRP",
                 "TYR",
                 "VAL",
                 "GLY",
                 ]

###################################################################
# # Autobuild constants
###################################################################
PANDDA_RHOFIT_SCRIPT_FILE = "pandda_rhofit.sh"
RHOFIT_COMMAND = (
    "#!/bin/bash \n"
    "source ~/.bashrc \n"
    "pandda_rhofit.sh -map {event_map} -mtz {mtz} -pdb {pdb} -cif {cif} -out {out_dir} -cut {cut}"
)
RHOFIT_COMMAND_COORD = (
    "#!/bin/bash \n"
    "source ~/.bashrc \n"
    "pandda_rhofit.sh -map {event_map} -mtz {mtz} -pdb {pdb} -cif {cif} -out {out_dir} -x {x} {y} {z} -cut {cut}"
)
MASKED_PDB_FILE = "masked.pdb"
TRUNCATED_EVENT_MAP_FILE = "truncated.ccp4"
CUT_EVENT_MAP_FILE = "cut.ccp4"
ELBOW_COMMAND = "cd {out_dir}; phenix.elbow {smiles_file} --output=\"{prefix}\"; cd -"
LIGAND_PREFIX = "ligand"
LIGAND_CIF_FILE = "ligand.cif"
LIGAND_PDB_FILE = "ligand.pdb"
GRADE_COMMAND = "cd {out_dir}; grade -in {smiles_file} -ocif {prefix}.cif -opdb {prefix}.pdb -f; cd -"
GRADE2_COMMAND = "cd {out_dir}; grade2 --in {smiles_file} -j -o {prefix} -f; mv {prefix}.restraints.cif {prefix}.cif; cd -"

###################################################################
# # Test constants
###################################################################

MISSES = [
    "BAZ2BA-x447",
    "BAZ2BA-x557",
    'ATAD2A-x1835',
    'CAMK1DA-x0083', 'CAMK1DA-x0091', 'CAMK1DA-x0145', 'CAMK1DA-x0208', 'CAMK1DA-x0218', 'CAMK1DA-x0299',
    'FAM83BA-x0786', 'FAM83BA-x0813', 'FAM83BA-x0837', 'FAM83BA-x0958', 'FAM83BA-x0963',
    'HAO1A-x0208', 'HAO1A-x0368', 'HAO1A-x0377', 'HAO1A-x0518', 'HAO1A-x0523', 'HAO1A-x0566', 'HAO1A-x0567',
    'HAO1A-x0592', 'HAO1A-x0599', 'HAO1A-x0603', 'HAO1A-x0671', 'HAO1A-x0678', 'HAO1A-x0713', 'HAO1A-x0763',
    'HAO1A-x0808', 'HAO1A-x0825', 'HAO1A-x0826', 'HAO1A-x0841', 'HAO1A-x0889',
    'ATAD2A-x1835',
    'CAMK1DA-x0091', 'CAMK1DA-x0208', 'CAMK1DA-x0218', 'CAMK1DA-x0299',
    'FAM83BA-x0786', 'FAM83BA-x0813', 'FAM83BA-x0837', 'FAM83BA-x0958', 'FAM83BA-x0963',
    'Mpro-x3305',
    'NUDT21A-x0750', 'NUDT21A-x0948', 'NUDT21A-x0967',
    'NUDT5A-x0158', 'NUDT5A-x0244', 'NUDT5A-x0256', 'NUDT5A-x0392', 'NUDT5A-x0574', 'NUDT5A-x0651', 'NUDT5A-x0681',
    'NUDT5A-x1083', 'NUDT5A-x1223', 'NUDT5A-x1235', 'NUDT5A-x1242',
    'PARP14A-x0412', 'PARP14A-x0473',
    'TBXTA-x0024', 'TBXTA-x0076', 'TBXTA-x0093', 'TBXTA-x0096', 'TBXTA-x0113', 'TBXTA-x0114', 'TBXTA-x0124',
    'TBXTA-x0128', 'TBXTA-x0144', 'TBXTA-x0164', 'TBXTA-x0167', 'TBXTA-x0179', 'TBXTA-x0204', 'TBXTA-x0229',
    'TBXTA-x0244', 'TBXTA-x0249', 'TBXTA-x0258', 'TBXTA-x0316', 'TBXTA-x0338', 'TBXTA-x0353', 'TBXTA-x0368',
    'TBXTA-x0390', 'TBXTA-x0409', 'TBXTA-x0464', 'TBXTA-x0465', 'TBXTA-x0491', 'TBXTA-x0500', 'TBXTA-x0506',
    'TBXTA-x0530', 'TBXTA-x0545', 'TBXTA-x0552', 'TBXTA-x0563', 'TBXTA-x0567', 'TBXTA-x0753', 'TBXTA-x0830',
    'TBXTA-x0835', 'TBXTA-x0938', 'TBXTA-x0948', 'TBXTA-x0992', 'TBXTA-x1079', 'TBXTA-x1161',
    'TNCA-x0146', 'TNCA-x0323',
    'XX02KALRNA-x1376', 'XX02KALRNA-x1377', 'XX02KALRNA-x1389', 'XX02KALRNA-x1463', 'XX02KALRNA-x1480',
    'XX02KALRNA-x1505',
    'DCLRE1AA-x0135', 'DCLRE1AA-x0169', 'DCLRE1AA-x0209', 'DCLRE1AA-x0965', 'DCLRE1AA-x0968', 'DCLRE1AA-x1015',
    'DCLRE1AA-x1074',
    'EPB41L3A-x0105', 'EPB41L3A-x0175', 'EPB41L3A-x0305', 'EPB41L3A-x0325', 'EPB41L3A-x0401', 'EPB41L3A-x0475',
    'EPB41L3A-x0520',
    'INPP5DA-x0020', 'INPP5DA-x0021', 'INPP5DA-x0027', 'INPP5DA-x0030', 'INPP5DA-x0039', 'INPP5DA-x0040',
    'INPP5DA-x0054', 'INPP5DA-x0083', 'INPP5DA-x0088', 'INPP5DA-x0106', 'INPP5DA-x0118', 'INPP5DA-x0119',
    'INPP5DA-x0126', 'INPP5DA-x0129', 'INPP5DA-x0133', 'INPP5DA-x0148', 'INPP5DA-x0163', 'INPP5DA-x0183',
    'INPP5DA-x0200', 'INPP5DA-x0221', 'INPP5DA-x0251', 'INPP5DA-x0300', 'INPP5DA-x0345', 'INPP5DA-x0359',
    'INPP5DA-x0382', 'INPP5DA-x0490', 'INPP5DA-x0560', 'INPP5DA-x0573', 'INPP5DA-x0577', 'INPP5DA-x0588',
    'INPP5DA-x0596',
    'OXA10OTA-x0051', 'OXA10OTA-x0059', 'OXA10OTA-x0079',
    'mArh-x0792', 'mArh-x0822', 'mArh-x0844', 'mArh-x0852', 'mArh-x0900', 'mArh-x0905', 'mArh-x0926', 'mArh-x0950',
    'mArh-x0967', 'mArh-x0969', 'mArh-x1018', 'mArh-x1092',
    'ATAD2A-x1835',
    'CAMK1DA-x0091', 'CAMK1DA-x0208', 'CAMK1DA-x0218', 'CAMK1DA-x0299',
    'FAM83BA-x0786', 'FAM83BA-x0813', 'FAM83BA-x0837', 'FAM83BA-x0958', 'FAM83BA-x0963',
    'Mpro-x10862', 'Mpro-x10889', 'Mpro-x1101',
    'NUDT21A-x0750', 'NUDT21A-x0948', 'NUDT21A-x0967',
    'NUDT5A-x0114', 'NUDT5A-x0125', 'NUDT5A-x0158', 'NUDT5A-x0169', 'NUDT5A-x0171', 'NUDT5A-x0176', 'NUDT5A-x0177',
    'NUDT5A-x0244', 'NUDT5A-x0256', 'NUDT5A-x0262', 'NUDT5A-x0286', 'NUDT5A-x0299', 'NUDT5A-x0319', 'NUDT5A-x0320',
    'NUDT5A-x0333', 'NUDT5A-x0339', 'NUDT5A-x0392', 'NUDT5A-x0403', 'NUDT5A-x0412', 'NUDT5A-x0463', 'NUDT5A-x0469',
    'NUDT5A-x0525', 'NUDT5A-x0526', 'NUDT5A-x0552', 'NUDT5A-x0554', 'NUDT5A-x0574', 'NUDT5A-x0600', 'NUDT5A-x0605',
    'NUDT5A-x0627', 'NUDT5A-x0637', 'NUDT5A-x0651', 'NUDT5A-x0673', 'NUDT5A-x0681', 'NUDT5A-x0685', 'NUDT5A-x0692',
    'NUDT5A-x1004', 'NUDT5A-x1024', 'NUDT5A-x1028', 'NUDT5A-x1083', 'NUDT5A-x1211', 'NUDT5A-x1235',
    'PARP14A-x0412', 'PARP14A-x0473',
    'TBXTA-x0113', 'TBXTA-x0164', 'TBXTA-x0204', 'TBXTA-x0229', 'TBXTA-x0244', 'TBXTA-x0258', 'TBXTA-x0338',
    'TBXTA-x0353', 'TBXTA-x0409', 'TBXTA-x0464', 'TBXTA-x0491', 'TBXTA-x0753', 'TBXTA-x0773', 'TBXTA-x0776',
    'TBXTA-x0786', 'TBXTA-x0820', 'TBXTA-x0830', 'TBXTA-x0835', 'TBXTA-x0938', 'TBXTA-x0948', 'TBXTA-x0986',
    'TBXTA-x0992', 'TBXTA-x1079', 'TBXTA-x1161', 'TBXTA-x1174',
    'TNCA-x0146', 'TNCA-x0192',
    'XX02KALRNA-x1376', 'XX02KALRNA-x1377', 'XX02KALRNA-x1388', 'XX02KALRNA-x1389', 'XX02KALRNA-x1449',
    'XX02KALRNA-x1453', 'XX02KALRNA-x1463', 'XX02KALRNA-x1480', 'XX02KALRNA-x1483', 'XX02KALRNA-x1490',
    'XX02KALRNA-x1492', 'XX02KALRNA-x1505',
    'DCLRE1AA-x0128', 'DCLRE1AA-x0135', 'DCLRE1AA-x0139', 'DCLRE1AA-x0150', 'DCLRE1AA-x0164', 'DCLRE1AA-x0169',
    'DCLRE1AA-x0183', 'DCLRE1AA-x0193', 'DCLRE1AA-x0194', 'DCLRE1AA-x0195', 'DCLRE1AA-x0209', 'DCLRE1AA-x0965',
    'DCLRE1AA-x0968', 'DCLRE1AA-x1015',
    'EPB41L3A-x0267', 'EPB41L3A-x0306', 'EPB41L3A-x0509', 'EPB41L3A-x0076', 'EPB41L3A-x0104', 'EPB41L3A-x0105',
    'EPB41L3A-x0141', 'EPB41L3A-x0150', 'EPB41L3A-x0161', 'EPB41L3A-x0162', 'EPB41L3A-x0171', 'EPB41L3A-x0175',
    'EPB41L3A-x0179', 'EPB41L3A-x0182', 'EPB41L3A-x0202', 'EPB41L3A-x0209', 'EPB41L3A-x0212', 'EPB41L3A-x0224',
    'EPB41L3A-x0235', 'EPB41L3A-x0244', 'EPB41L3A-x0249', 'EPB41L3A-x0274', 'EPB41L3A-x0305', 'EPB41L3A-x0316',
    'EPB41L3A-x0325', 'EPB41L3A-x0337', 'EPB41L3A-x0401', 'EPB41L3A-x0403', 'EPB41L3A-x0413', 'EPB41L3A-x0419',
    'EPB41L3A-x0429', 'EPB41L3A-x0455', 'EPB41L3A-x0457', 'EPB41L3A-x0463', 'EPB41L3A-x0473', 'EPB41L3A-x0475',
    'EPB41L3A-x0485', 'EPB41L3A-x0494', 'EPB41L3A-x0520', 'EPB41L3A-x0527', 'EPB41L3A-x0538', 'EPB41L3A-x0544',
    'EPB41L3A-x0547', 'EPB41L3A-x0550', 'EPB41L3A-x0556', 'EPB41L3A-x0558', 'EPB41L3A-x0575', 'EPB41L3A-x0598',
    'EPB41L3A-x0601', 'EPB41L3A-x0608', 'EPB41L3A-x0630',
    'INPP5DA-x0019', 'INPP5DA-x0020', 'INPP5DA-x0021', 'INPP5DA-x0027', 'INPP5DA-x0030', 'INPP5DA-x0039',
    'INPP5DA-x0040', 'INPP5DA-x0054', 'INPP5DA-x0058', 'INPP5DA-x0062', 'INPP5DA-x0067', 'INPP5DA-x0073',
    'INPP5DA-x0075', 'INPP5DA-x0083', 'INPP5DA-x0084', 'INPP5DA-x0088', 'INPP5DA-x0092', 'INPP5DA-x0097',
    'INPP5DA-x0100', 'INPP5DA-x0101', 'INPP5DA-x0103', 'INPP5DA-x0106', 'INPP5DA-x0114', 'INPP5DA-x0118',
    'INPP5DA-x0119', 'INPP5DA-x0121', 'INPP5DA-x0126', 'INPP5DA-x0129', 'INPP5DA-x0133', 'INPP5DA-x0140',
    'INPP5DA-x0141', 'INPP5DA-x0148', 'INPP5DA-x0152', 'INPP5DA-x0154', 'INPP5DA-x0161', 'INPP5DA-x0163',
    'INPP5DA-x0169', 'INPP5DA-x0174', 'INPP5DA-x0176', 'INPP5DA-x0182', 'INPP5DA-x0183', 'INPP5DA-x0194',
    'INPP5DA-x0195', 'INPP5DA-x0200', 'INPP5DA-x0201', 'INPP5DA-x0221', 'INPP5DA-x0239', 'INPP5DA-x0243',
    'INPP5DA-x0251', 'INPP5DA-x0300', 'INPP5DA-x0302', 'INPP5DA-x0303', 'INPP5DA-x0311', 'INPP5DA-x0316',
    'INPP5DA-x0320', 'INPP5DA-x0325', 'INPP5DA-x0340', 'INPP5DA-x0342', 'INPP5DA-x0345', 'INPP5DA-x0350',
    'INPP5DA-x0358', 'INPP5DA-x0359', 'INPP5DA-x0361', 'INPP5DA-x0368', 'INPP5DA-x0378', 'INPP5DA-x0382',
    'INPP5DA-x0401', 'INPP5DA-x0403', 'INPP5DA-x0417', 'INPP5DA-x0423', 'INPP5DA-x0438', 'INPP5DA-x0449',
    'INPP5DA-x0469', 'INPP5DA-x0479', 'INPP5DA-x0490', 'INPP5DA-x0494', 'INPP5DA-x0510', 'INPP5DA-x0518',
    'INPP5DA-x0524', 'INPP5DA-x0527', 'INPP5DA-x0542', 'INPP5DA-x0543', 'INPP5DA-x0558', 'INPP5DA-x0560',
    'INPP5DA-x0573', 'INPP5DA-x0577', 'INPP5DA-x0588', 'INPP5DA-x0590', 'INPP5DA-x0596', 'INPP5DA-x0604',
    'INPP5DA-x0609',
    'OXA10OTA-x0051', 'OXA10OTA-x0079',
    'mArh-x0091', 'mArh-x0792', 'mArh-x0822', 'mArh-x0844', 'mArh-x0852', 'mArh-x0900', 'mArh-x0905', 'mArh-x0926',
    'mArh-x0950', 'mArh-x0967', 'mArh-x0969', 'mArh-x1018', 'mArh-x1092',
]
