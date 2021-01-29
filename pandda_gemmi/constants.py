STRIPPED_RECEPTOR_FILE = "stripped_receptor.pdb"
LIGAND_FILE = "autobuilding_ligand.cif"
EVENT_MTZ_FILE = "event.mtz"
GRAFTED_MTZ_FILE = "grafted.mtz"
RHOFIT_DIR = "rhofit"
RHOFIT_EVENT_DIR = "rhofit_{}"
RHOFIT_NORMAL_DIR = "rhofit_normal"
RHOFIT_RESULTS_FILE = "results.txt"
RHOFIT_RESULT_JSON_FILE = "result.json"
RHOFIT_NORMAL_RESULT_JSON_FILE = "result_normal.json"
RHOFIT_BEST_MODEL_FILE = "best.pdb"
RSCC_TABLE_FILE = "rscc_table.csv"

BUILD_DIR_PATTERN = "{pandda_name}_{dtag}_{event_idx}"


CUMULATIVE_HITS_PLOT_FILE = "cumulative_hits.png"

PANDDA_ANALYSES_DIR = "analyses"
PANDDA_ANALYSE_EVENTS_FILE = "pandda_analyse_events.csv"
PANDDA_ANALYSE_SITES_FILE = "pandda_analyse_sites.csv"
PANDDA_PROCESSED_DATASETS_DIR = "processed_datasets"
PANDDA_MODELLED_STRUCTURES_DIR = "modelled_structures"
PANDDA_LIGAND_FILES_DIR = "ligand_files"
PANDDA_PDB_FILE = "{}-pandda-input.pdb"
PANDDA_MTZ_FILE = "{}-pandda-input.mtz"

PANDDA_LIGAND_CIF_FILE = "ligand.cif"
PANDDA_LIGAND_PDB_FILE = "ligand.pdb"

PANDDA_INSPECT_EVENTS_PATH = "pandda_inspect_events.csv"
PANDDA_EVENT_MAP_FILE = "{}-event_{}_1-BDC_{}_map.native.ccp4"
PANDDA_EVENT_MODEL = "{}-pandda-model.pdb"

PANDDA_PROTEIN_MASK_FILE = "protein_mask.ccp4"
PANDDA_SYMMETRY_MASK_FILE = "symmetry_mask.ccp4"
PANDDA_TOTAL_MASK_FILE = "total_mask.ccp4"
PANDDA_MEAN_MAP_FILE = "mean_{number}_{res}.ccp4"
PANDDA_SIGMA_S_M_FILE = "sigma_s_m_{number}_{res}.ccp4"

PANDDA_Z_MAP_FILE = "{dtag}-z_map.native.ccp4"
PANDDA_EVENT_MAP_FILE = "{dtag}-event_{event_idx}_1-BDC_{bdc}_map.native.ccp4"

PANDDA_LOG_FILE = "pandda_log.json"


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
]

STAGE_FILTER_INVALID = "Filter invalid dataset"
STAGE_FILTER_LOW_RESOLUTION = "Filter low resolution datasets"
STAGE_FILTER_RFREE = "Filter bad rfree dataset"
STAGE_FILTER_WILSON = "Filter bad wilson datasets"
STAGE_FILTER_STRUCTURE = "Filter dissimilar structure datasets"
STAGE_FILTER_SPACE_GROUP = "Filter different space group datasets"
STAGE_FILTER_GAPS = "Filter datasets with large gaps"


MISSES = [
    'ATAD2A-x1835',
    'CAMK1DA-x0083', 'CAMK1DA-x0091', 'CAMK1DA-x0145', 'CAMK1DA-x0208', 'CAMK1DA-x0218', 'CAMK1DA-x0299',
    'FAM83BA-x0786', 'FAM83BA-x0813', 'FAM83BA-x0837', 'FAM83BA-x0958', 'FAM83BA-x0963',
    'HAO1A-x0208', 'HAO1A-x0368', 'HAO1A-x0377', 'HAO1A-x0518', 'HAO1A-x0523', 'HAO1A-x0566', 'HAO1A-x0567', 'HAO1A-x0592', 'HAO1A-x0599', 'HAO1A-x0603', 'HAO1A-x0671', 'HAO1A-x0678', 'HAO1A-x0713', 'HAO1A-x0763', 'HAO1A-x0808', 'HAO1A-x0825', 'HAO1A-x0826', 'HAO1A-x0841', 'HAO1A-x0889',
    ]