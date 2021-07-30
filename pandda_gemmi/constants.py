###################################################################
# # Pandda File names
###################################################################

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
    ("2FOFCWT", "2PHFOFCWT"),
    ("2FOFCWT_iso-fill", "PH2FOFCWT_iso-fill"),
    ("2FOFCWT_fill", "PH2FOFCWT_fill",),
)

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

LOG_DATASETS: str = "Summary of input datasets"

LOG_SHELL_XMAP_TIME: str = "Time taken to generate aligned xmaps"
LOG_SHELL_DATASET_LOGS: str = "Logs for each dataset in shell"
LOG_SHELL_TIME: str = "Time taken to process shell"
LOG_SHELLS: str = "Logs for each shell"

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
MASKED_PDB_FILE = "masked.pdb"
TRUNCATED_EVENT_MAP_FILE = "truncated.ccp4"
CUT_EVENT_MAP_FILE = "cut.ccp4"
ELBOW_COMMAND = "cd {out_dir}; phenix.elbow {smiles_file} --output=\"{prefix}\"; cd -"
LIGAND_PREFIX = "ligand"
LIGAND_CIF_FILE = "ligand.cif"

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
