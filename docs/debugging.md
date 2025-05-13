# PanDDA 2 User Guide

 - [How PanDDA 2 Works](#how-pandda-2-works)
 - [Using PanDDA 2](#using-pandda-2)
 - [FAQ And Debugging](#faqs-and-debugging)


### How PanDDA 2 Works

PanDDA 2 differs from PanDDA 1 in two major ways: 

1. Firstly, it attempts to identify which sets of datasets should be used to produce a statistical model which has the optimal contrast (to each test dataset individually). This allows it to handle subtle heterogeneity between datasets. 

2. Secondly it attempts to autobuild the events returned, and then rank the events based on the quality of the model of the fragment that could be constructed. This allows improved rankings of events.

### How PanDDA 2 Characterizes Ground States

PanDDA 2 differs significantly from PanDDA 1 in how it detects outlying density. 

 - **PanDDA 1**: Establishes the expected electron density around a single reference protein model from the first N datasets above a set of resolution cutoffs

 - **PanDDA 2**: Establishes several difference possibilities for the expected electron density around each dataset based on a several sets of similar datasets above the resolution of that dataset

The difference in how the ground state is determined has significant concequences:
1. PanDDA 2 generates FAR more events, at least before filters, because it evaluates multiple statistical models, some of which will generate more events than PanDDA 1's statistical model because they describe the density poorly.
2. PanDDA 2 able to detect extremely weak binding and handle hetrogenous crystal systems, but also means that it is reliant on good filtering of events in order to run quickly and present a reasonable amount of data to the user.
3. It does not make sense to talk of a "reference" dataset in PanDDA 2: each dataset is effectively its own reference, in PanDDA 1 terminology.


### Statistical Model Selection

Currently PanDDA 2 only chooses one ground state to progress to event map generation and autobuilding. This is done by fitting several conformations of the expected fragment into each event from each statistical by differential evolution. 

The ground state which has the best fragment fit is then selected for progression to event map generation and autobuilding.

### Autobuilding

PanDDA 2 autobuilds the events of each dataset. This is done with a custom method that uses differential evolution to fit multiple conformations of the fragment to outlying density.

### Autobuild Selection

Current limitations with the interaction between `pandda.inspect` and PanDDA 2 mean that it is only possible to show one autobuild in the GUI. 

Therefore, the highest scoring autobuild by build score from any event in each dataset is selected and included in the initial model `pandda.inspect` shows the user. 

This has the effect that users may open apparently good hit density with no autobuild present, if another hit which is better fit by the autobuilding is present in the same dataset.






## Using PanDDA 2

### Minimum Requirements For Good PanDDA Performance

PanDDA 2 is able to find low occupancy ligands by combining information from multiple crystallographic datasets. As such having a set of near-homogenous crystal structures is required. Although sufficient homogeneity for good PanDDA 2 results varies, good guidelines are:

1. The same spacegroup
2. Unit cell parameters within ~10% of one another
3. Structure RMSDs that are _generally_ less than 2A RMSD, although regions of local heterogeneity are generally well tolerated by the algorithm

It is important to note that PanDDA 2 can automatically handle _multiple_ sets of near-homogenous crystal structures at the same time i.e. two groups of datasets that do not meet the above criteria. Indeed, there may be advantages to processing both groups at the same time as one set may provide a better contrast to ligand bound changed states in the other.

In particular, for good results the minimum requirements are:
1. A minimum of 60 near-homogenous datasets in total. 
2. For each binding site, at least 30 datasets that do not contain ligands bound there. This is generally not possible to guarantee ahead of PanDDA analysis, however if the total number of datasets is greater than 100, with the typical fragment screen hit rates of ~15%, predominantly in a small number of hotspots, this is likely. This factor may also diagnose poor results from relatively screens.
3. A relatively high resolution (in practice over 80% of hits found at XChem are discovered in crystal structures with a resolution better than ~2.1 Angstrom)

Of course, every crystal system is different, and depending upon the exact data being analyzed, good results may be possible without meeting any or even all of these requirements. However, this will likely require manually specifying significant numbers of parameters and a strong understanding of both the crystal system and PanDDA methodology. In general, 200-300 datasets should be sufficient for very good results in even relatively heterogeneous systems with default settings.

### The PanDDA Input Data File Structure

PanDDA expects input data with the following structure:

```text

<Directory matching option --data_dirs>
├── <Crystal Name 1>
│   ├── ...
...
├── <Crystal Name N>
│   ├── compound
│   │   ├── <ligand CIF file with same file stem as ligand PDB>
│   │   ├── <ligand PDB file with same file stem as ligand CIF>
│   ├── <PDB file matching option --pdb_regex>
│   ├── <MTZ file mathcing option --mtz_regex>
...

```

For example:

```text
{using --pdb_regex="dimple.pdb" --mtz_regex="dimple.mtz"}

model_building/
├── D68EV3CPROA-x0001
│   ├── ...
...
├── D68EV3CPROA-x0110
│   ├── compound
│   │   ├── Z104924088.cif
│   │   ├── Z104924088.pdb
│   ├── dimple.mtz -> dimple/dimple/final.mtz
│   ├── dimple.pdb -> dimple/dimple/final.pdb
...

```

### Common Options And Their Uses

PanDDA 2 has a number of common command line options that may be of interest to users. Here are the most common with links to how to use them.

- `--dataset_range`
  - [How To Run Subsets Of Data](#how-to-run-subsets-of-data)
- `--only_datasets`
  - [How To Run Subsets Of Data](#how-to-run-subsets-of-data)
- `--high_res_lower_limit`
  - [How To Filter Poor Quality Data](#how-to-filter-poor-quality-data)
- `--max_rfree`
  - [How To Filter Poor Quality Data](#how-to-filter-poor-quality-data)

### How To Run Subsets Of Data

It is often useful to only analyze a subset of data, for example because new data has become available and is unlikely to improve old results, or because only some datasets are of interest. It is important to note that none of these options prevent datasets being used to characterize ground state models, it only prevents events and autobuilds being generated for them.

PanDDA 2 provides several options that allow users to do this:
1. `--dataset_range`: Process only those datasets whose names (the name of the directory containing their data) end in numbers been the two bounds. An example might be `--dataset_range="100-200"`, which will process `BAZ2BA-x0102`, but not `BAZ2BA-x0097`.
2. `--exclude_from_z_map_analysis`: Process only those datasets whose names do not occur in the list. For example, `--exclude_from_z_map_analysis="BAZ2BA-x102,BAZ2BA-x097"` will not process `BAZ2BA-x102`, but will process `BAZ2BA-x033`.
3. `--only_datasets`: Only process those datasets whose names occur in the given string. For example, `--only_datasets="BAZ2BA-x088,BAZ2BA-x092"` will process `BAZ2BA-x088` but will not process `BAZ2BA-x105`.

These options are not exclusive, and are applied in the above order, so if `--dataset_range="100-200"` and `--only_datasets="BAZ2BA-x157"` the **no** datasets will be processed.

### How To Filter Poor Quality Data

Although the defaults for PanDDA are usually appropriate, sometimes it is necessary change the defaults on what constitues sufficiently good dataset to use in ground state characterization.

The main options PanDDA 2 provides for this are:
1. `--high_res_lower_limit`: Process only those datasets whose assigned high resolution limit is above this value. For example, `--high_res_lower_limit=3.0` will only process those datasets whose upper resolution limit is between 0.0 and 3.0.
2. `--max_rfree`: Process only those datasets whose assigned rfree is below this value. For example, `--max_rfree=3.0` will only process those datasets whose rfree is between 0 and 3.0.

### How to recover from Failed Runs

If PanDDA 2 has begun processing datasets but ends before it completes, for example because it was cancelled by a cluster scheduler, then it is possible to continue from the last dataset that completed processing.

This is as simple as running the same PanDDA command that began the run again. The program will detect that processed datasets are present and then continue.

A warning with this feature is that if the input data has changed, new datasets will be processed against the current data, not the original data. This could happen if, for example, new datasets have been collected.

(Diamond users can rerun pandda2 targeting the same directory as the failed run through XChemExplorer as if they were running a new PanDDA there).

### How To Run PanDDA 2 Against Crude Soaks And Cocktails

If you have a crude or other cocktail experiment in which multiple ligands have been soaked, then it is important to make sure PanDDA has data for all the ligands that may have bound.

This requires no special configuration of PanDDA 2, which will automatically handle this case if it occurs, it just requires making sure that the data is available. For example, had multiple ligands been soaked against the example dataset given [above](#the-pandda-input-data-file-structure), it might look like this:

```text

model_building/
├── D68EV3CPROA-x0001
...
├── D68EV3CPROA-x0110
│   ├── compound
│   │   ├── Z104924088.cif
│   │   ├── Z104924088.pdb
│   │   ├── some_other_ligand.cif
│   │   ├── some_other_ligand.pdb
│   ├── dimple.mtz -> dimple/dimple/final.mtz
│   ├── dimple.pdb -> dimple/dimple/final.pdb
...

```



### Discovering Which Datasets Were Used to Characterize the Ground State

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

### Interpreting PanDDA 2 Event Maps and Z-Maps

It is important to explore the fit of ligands at multiple contours in PanDDA Event Maps and Z-maps in order to determine whether the evidence for binding is robust. Electron Densities in PanDDA event maps in particular are unlikely to mean the same thing as they would in a conventional 2Fo-Fc map due to the rescaling of features in the event map calculation, and hence the quality of the fit is best determined by how well the expected "shape"/"topology"" of the ligand is reproduced at contour for which this is is best reproduced rather than pre-defining some level which represents significance. 

It is also important to consider whether binding is likely to be driven by crystal packing. While to some extent this is an art, because fragments which form small numbers of interactions with symmetry artefacts have proven useful in medicinal chemsity, a feeling for whether fragments are likely to pose a risk of being crystallographic artefacts can be established by looking at the relative number of interactions formed with artefact atoms versus non-artefact atoms. Ideally there should be zero interactions with crystallographic artefacts.

It is important to know your crystal system: symmetry atoms _may_ be a part of a biological assembly, for example at the interface between the two protein chains of a dimer in which only one chain is in the ASU. In this case forming interactions with the symmetry atoms is likely to be a positive sign rather than a warning sign.

In `pandda.inspect`, the easiest way to determine this is by finding "Draw -> Cell and Symmetry" in the command bar and ensuring the "Master Switch" is "Yes". Then going to the command bar and selecting "Measures -> Environment Distances" and ensuring that "Show Residue Environment" is ticked will highlight likely bonds.

### Determining the reasons datasets were not analyzed or included in ground states

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

## FAQs and Debugging


### PanDDA 2 has returned an enourmous number of events
This is working as intended: PanDDA 2 is designed re return significantly more events than PanDDA 1, but to rank them better. 

Events are ordered within each site by their build score, so if necessary one can stop analysing each site after several non-hits have been observed.

### The event map doesn't resemble the protein

This is to be expected if the event map's event does not is not a fragment or actual meaningful change: in such cases the event map is effectively random and should be ignored. An example of this is shown below:

![low_ranking_event](https://github.com/ConorFWild/pandda_2_gemmi/raw/0_1_0/imgs/low_ranking_event.png)


If there is clearly a fragment present but the event map's quality seems low for the protein: this is typically due to a poorly characterized ground state model and such event maps can be used as normal: i.e. can be built into. An example can be seen below:

### An event's density looks like the soaked fragment but it has not been autobuilt

There are two possibilities. The first is that autobuilding the event has failed, typically because the outlying Z-values form a small blob.

The second possibility is that the event has been autobuilt, but PanDDA prefers another event in the protein to merge into the model shown in `pandda.inspect`. This is because in the current version of PanDDA inspect only one autobuild can be chosen to display for any dataset, regarless of how many events it has. PanDDA does occasionally chose a poor autobuild over a good one, so it is important in such cases to manually build the good density and then check that if PanDDA has placed a fragment it too is in a sensible place.

### Many datasets don't seem to have any event maps

It is important to check the PanDDA logs to make sure that datasets you are interested in have not been filtered from consideration. See [here](#determining-the-reasons-datasets-were-not-analyzed-or-included-in-ground-states).

### An event has been autobuilt but incorrectly

Autobuilding PanDDA events is not perfect and the method can sometimes fail. If you see an event in `pandda.inspect` that appears to resemble the fragment but the autobuild is poor, the advice is to correct the autobuild and save the new model.

PanDDA 2 can sometimes fail to produce good autobuilds of clear events. While by far the most likely cause of this was a failure in selecting the best build, which is an area of active development, there are some other causes as well:

1. If the ligand is very large or flexible, the number of conformations to sampled may be too low to get one close to the crystallographic conformation
2. There is unusual Z-map density at or near the ligand

### Handeling very large numbers of spurious events

Depending on your target, PanDDA 2 may produce very large numbers of events. Although PanDDA attempts to rank and classify these, in some scenarios it may be preferable to risk losing some hits with weaker evidence in order to make processing faster. 

This can be achieved in several ways:
1. Decrease the number of events processed using the option `--event_score_threshold`
2. Change the max events per dataset using the option `--max_events_per_dataset`

### Addressing very long run times

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

### A PanDDA job was cancelled

If a long-running PanDDA job was cancelled, for example by the cluster or a machine failure, it can be easily resumed from where it left off. See [here](#how-to-recover-from-failed-runs).

### Ligand Electron Density Is Partially Missing

PanDDA masks the electron density around the chains contained in the supplied PDB files. While this is useful for a number of reasons, such as decreasing run times and limiting the number of spurious events, this can sometimes result in partial density for ligands which stick out into solvent.

If this is observed the reccomended course of action is to collate all datasets which feature such ligands, and then rerun PanDDA in a new directory with the `--only_datasets` option set to the datasets containing these ligands and the `--outer_mask` option set to 9.0+(the distance in Angstron between the defined region of them map and the furthest ligand atom).

