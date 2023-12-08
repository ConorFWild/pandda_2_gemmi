# Debugging PanDDA 2

## Minimum Requirements for Good PanDDA Performance


## The PanDDA Input Data File Structure

## How to run subsets of data


## How to recover from Failed Runs

If PanDDA 2 has begun processing datasets but ends before it completes, for example because it was cancelled by a cluster scheduler, then it is possible to continue from the last dataset that completed processing.

This is as simple as running the same PanDDA command that began the run again. The program will detect that processed datasets are present and then continue.

A warning with this feature is that if the input data has changed, new datasets will be processed against the current data, not the original data. This could happen if, for example, new datasets have been collected.

## How PanDDA 2 Characterizes Ground States

PanDDA 2 differs significantly from PanDDA 1 in how it detects outlying density. 

 - **PanDDA 1**: Establishes the expected electron density around a single reference protein model from the first N datasets above a set of resolution cutoffs

 - **PanDDA 2**: Establishes several difference possibilities for the expected electron density around each dataset based on a several sets of similar datasets above the resolution of that dataset

The difference in how the ground state is determined has significant concequences:
1. PanDDA 2 generates FAR more events, at least before filters, because it evaluates multiple statistical models, some of which will generate more events than PanDDA 1's statistical model because they describe the density poorly.
2. PanDDA 2 able to detect extremely weak binding and handle hetrogenous crystal systems, but also means that it is reliant on good filtering of events in order to run quickly and present a reasonable amount of data to the user.
3. It does not make sense to talk of a "reference" dataset in PanDDA 2: each dataset is effectively its own reference, in PanDDA 1 terminology

## Discovering Which Datasets Were Used to Characterize the Ground State

The information on which datasets were used to characterize the statistical model from which the Z-map, events and event maps can be helpful in determining whether sensible ground state models were used or explaining anomalies in Z-maps or mean maps. This information can be found in two places:

1. The Console Output: Under each dataset's subheading in the console output it a section for which datasets were used for each statistical model, and which statistical model was selected.

```text
╭──────────────────────────────────────────────────────────────────────────────╮
│                             Mpro-x0072 : 1 / 54                              │
╰──────────────────────────────────────────────────────────────────────────────╯
    Resolution: 1.65
    Processing Resolution: 1.65
───────────────────────────── Comparator Datasets ──────────────────────────────
    Mpro-x0014 Mpro-x0017 Mpro-x0019 Mpro-x0022 Mpro-x0025 Mpro-x0026 Mpro-x0027
    Mpro-x0030 Mpro-x0040 Mpro-x0041 Mpro-x0058 Mpro-x0059 Mpro-x0072 Mpro-x0074
    Mpro-x0090 Mpro-x0091 Mpro-x0103 Mpro-x0104 Mpro-x0105 Mpro-x0113 Mpro-x0114

...

────────────────────────────── Model Information ───────────────────────────────
    Processed Models: [10, 7, 9]
    Selected model: 10
    Model Number: 1
        Processed: False
        Mpro-x2041 Mpro-x2110 Mpro-x2139 Mpro-x2903 Mpro-x2967 Mpro-x2976
        Mpro-x2996 Mpro-x3002 Mpro-x3004 Mpro-x3015 Mpro-x3037 Mpro-x3071
        Mpro-x3072 Mpro-x3073 Mpro-x3102 Mpro-x3104 Mpro-x3105 Mpro-x3118
        Mpro-x3121 Mpro-x3129 Mpro-x3145 Mpro-x3151 Mpro-x3173 Mpro-x3215
        Mpro-x3255
...
    Model Number: 7
        Processed: True
        Mpro-x0025 Mpro-x0103 Mpro-x0153 Mpro-x0222 Mpro-x0227 Mpro-x0242
        Mpro-x0263 Mpro-x0271 Mpro-x0278 Mpro-x0295 Mpro-x0302 Mpro-x0313
        Mpro-x0340 Mpro-x0343 Mpro-x0392 Mpro-x0397 Mpro-x0480 Mpro-x0523
        Mpro-x0997 Mpro-x1031 Mpro-x1151 Mpro-x1245 Mpro-x1266 Mpro-x1288
        Mpro-x2150
...
    Model Number: 10
        Processed: True
        Mpro-x0014 Mpro-x0017 Mpro-x0019 Mpro-x0022 Mpro-x0025 Mpro-x0026
        Mpro-x0027 Mpro-x0030 Mpro-x0040 Mpro-x0041 Mpro-x0058 Mpro-x0059
        Mpro-x0072 Mpro-x0074 Mpro-x0090 Mpro-x0091 Mpro-x0103 Mpro-x0104
        Mpro-x0105 Mpro-x0113 Mpro-x0114 Mpro-x0150 Mpro-x0153 Mpro-x0174
        Mpro-x0221 Mpro-x0222 Mpro-x0227 Mpro-x0231 Mpro-x0240 Mpro-x0242
─────────────────────────── Processed Model Results ────────────────────────────
    Model Number: 10
        Number of events: 5
    Model Number: 7
        Number of events: 2
    Model Number: 9
        Number of events: 3


```
2. The Processed Dataset yaml: Located in <pandda output directory>/processed_datasets/<dataset name>/processed_dataset.yaml, this file contains information on which datasets were used and which model was selected. This file also contains additional information such as the score of the model and details of the events.

```yaml
Summary:
  Processing Resolution: 1.65
  Comparator Datasets:
  - Mpro-x0014
  - Mpro-x0017
  - Mpro-x0019
...
  Selected Model: 10
  Selected Model Events:
  - 1
  - 2
Models:
  1:
    Processed?: false
    Characterization Datasets:
    - Mpro-x2041
    - Mpro-x3072
    - ...
    - Mpro-x2903
    - Mpro-x3118
    Model Score: 0.46
    Events: {}
...
  7:
    Processed?: true
    Characterization Datasets:
    - Mpro-x0392
    - Mpro-x0103
...
    - Mpro-x0271
    - Mpro-x0278
    Model Score: 0.26
    Events:
      1:
        Score: 0.77762371301651
        BDC: 0.95
        Size: 5.216026807289329
        Centroid:
        - 13.797403329950091
        - 31.124850986842112
        - 5.65212530923325
        score: 0.593933344202621
        Local Strength: 24.15843994955058
        Build Score: -175.38088989257812
        Noise: 1118.583251953125
        Signal: 664.3638916015625
        Num. Contacts: 12
        Num. Points: 357.0
        Optimal Contour: 5.358448997892515
        RSCC: 0.27030460363399444
        Ligand Centroid:
        - 13.431249381912282
        - 29.929987307326126
        - 5.593201194030271
...

```

## Interpreting PanDDA 2 Event Maps and Z-Maps

It is important to explore the fit of ligands at multiple contours in PanDDA Event Maps and Z-maps in order to determine whether the evidence for binding is robust. Electron Densities in PanDDA event maps in particular are unlikely to mean the same thing as they would in a conventional 2Fo-Fc map due to the rescaling of features in the event map calculation, and hence the quality of the fit is best determined by how well the expected shape of the ligand is reproduced at contour for which this is is best reproduced rather than pre-defining some level which represents significance. 

1.2V
0.8V

It is also important to consider whether binding is likely to be driven by crystal packing. While to some extent this is an art, because fragments which form small numbers of interactions with symmetry artefacts have proven useful in medicinal chemsity, a feeling for whether fragments are likely to pose a risk of being crystallographic artefacts can be established by looking at the relative number of interactions formed with artefact atoms versus non-artefact atoms.

It is important to know your crystal system: symmetry atoms _may_ be a part of a biological assembly, for example at the interface between the two protein chains of a dimer in which only one chain is in the ASU. In this case forming interactions with the symmetry atoms is likely to be a positive sign rather than a warning sign.

In `pandda.inspect`, the easiest way to determine this is by finding "Draw -> Cell and Symmetry" in the command bar and ensuring the "Master Switch" is "Yes". Then going to the command bar and selecting "Measures -> Environment Distances" and ensuring that "Show Residue Environment" is ticked will highlight likely bonds.

## Determining the reasons datasets were not analyzed or included in ground states

PanDDA 2 will not analyse datasets for several reasons:
1. The RFree is > 0.4
2. The dataset were outside the range defined in the option `--dataset_range`
3. The datasets were in the excluded datasets defined in the option `--exclude_from_z_map_analysis`
4. The datasets were not in the datasets to be analysed defined in the option `--only_datasets`
5. The datasets did not have a ligand file with a .cif and .pdb in the directory given by the option `--compound_dir`
6. The datasets did not have a valid protein pdb or reflections mtz, given by the options `--pdb_regex` and `--mtz_regez` respectively
7. The ligand cif contains a delocalized system that PanDDA 2 cannot handle


```text
╭──────────────────────────────────────────────────────────────────────────────╮
│                             Loading datasets...                              │
╰──────────────────────────────────────────────────────────────────────────────╯
...
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Dtag                 ┃ Resolution   ┃ Spacegroup  ┃ SMILES?  ┃ CIF?  ┃ PDB?  ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ D68EV3CPROA-x0059    │ 1.56         │ P 21 21 21  │ True     │ False │ False │
│ D68EV3CPROA-x0060    │ 1.5          │ P 21 21 21  │ True     │ False │ False │
│ D68EV3CPROA-x0061    │ 1.68         │ P 21 21 21  │ True     │ False │ False │
│ D68EV3CPROA-x0062    │ 1.95         │ P 21 21 21  │ False    │ False │ False │
│ D68EV3CPROA-x0063    │ 1.52         │ P 21 21 21  │ False    │ False │ False │

```

```text
╭──────────────────────────────────────────────────────────────────────────────╮
│                       Building model of file system...                       │
╰──────────────────────────────────────────────────────────────────────────────╯
...
NO PDBS IN DIR: /dls/science/groups/i04-1/conor_dev/baz2b_test/data/coot-backup WITH REGEX: *dimple.pdb
NO MTZS IN DIR: /dls/science/groups/i04-1/conor_dev/baz2b_test/data/coot-backup WITH REGEX: *dimple.mtz

```

```text
╭──────────────────────────────────────────────────────────────────────────────╮
│                             Loading datasets...                              │
╰──────────────────────────────────────────────────────────────────────────────╯
                              Unit Cell Statistics
...
─────────────────────────── Datasets not to Process ────────────────────────────
    BAZ2BA-x425 : Filtered because in exclude set!
...
    BAZ2BA-x513 : Filtered because rfree > 0.4

```

A note on delocalized systems: PanDDA 2 uses RDKit to process ligands, which cannot handle non-atomatic delocalized systems such as sulfonates. Some cif building systems will mark these bonds correctly as delocalized, which will mean that PanDDA will not be able to kekulize them for processing. In order for PanDDA to process them you MUST ENSURE that delocalized systems are properly kekulized, which can be achieved by using a cif program such as GRADE.

A good give-away that this has happened is that datasets within the screen that you expect to have cifs are missing them.
```text
╭──────────────────────────────────────────────────────────────────────────────╮
│                             Loading datasets...                              │
╰──────────────────────────────────────────────────────────────────────────────╯
...
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Dtag                 ┃ Resolution   ┃ Spacegroup  ┃ SMILES?  ┃ CIF?  ┃ PDB?  ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ D68EV3CPROA-x0059    │ 1.56         │ P 21 21 21  │ True     │ False │ False │
...
│ D68EV3CPROA-x1799    │ 1.47         │ P 21 21 21  │ False    │ False │ False │
...
```


## Handeling very large numbers of spurious events

Depending on your target, PanDDA 2 may produce very large numbers of events. Although PanDDA attempts to rank and classify these, in some scenarios it may be preferable to risk losing some hits with weaker evidence in order to make processing faster. 

This can be achieved in several ways:
1. Decrease the number of events processed using the option `--event_score_threshold`
2. Change the max events per dataset using the option `--max_events_per_dataset`
3. 

## Addressing very long run times

If PanDDA 2 is taking a very long time to run it may be for one of several reasons:
1. Very large numbers of events have been detected
2. The ligands soaked are very large and/or flexible
3. The ASU is large and/or data is high resolution
4. A small number of cores are used for processing
5. There are many datasets
6. There are multiple ligands per dataset (for example in a crude screen)

If faster run times are necessary, then there are a few possible solutions:
1. Use more cores, which can be controlled with the `--local_cpus` option
2. Split up the PanDDA and run on multiple nodes, by launching several PanDDAs with different settings for the `--dataset_range` options, and then merging them with the script `scripts/merge_panddas.py`
3. Increase the threshold for evidence to process an event using the option `--process_event_score_cutoff`
4. Decrease the number of conformations tested using the option `--autobuild_max_num_conformations`
5. Decrease the number of autobuild attempts using the option `--autobuild_max_tries`

## PanDDA 2 fails to produce good autobuilds of clear events

PanDDA 2 can sometimes fail to produce good autobuilds of clear events. While by far the most likely cause of this was a failure in selecting the best build, which is an area of active development, there are some other causes as well.

1. If the ligand is very large or flexible, the number of conformations to sampled may be too low to get one close to the crystallographic conformation



Additional things to document:
1. Highlight causes of datasets being thrown out, and where in the log to find that
2. Highlight mask shell and how this can cut off density (given chains, not NCS ops)
3. Throw too many events and how to change the number
4. More things on how autobuilding works? Where it might go wrong (num poses, wrong selected ligand, expected performance)
5. Info on how to get good results for datasets
6. Running subsets (how to)
7. PanDDA file structure
8. Autobuilding multiple cifs (crudes)
9. Default cutoffs / minimal defaults for running