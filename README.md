# PanDDA 2

## Acknowledgements

## User Guide
[User Guide](./docs/debugging.md)

## New in version 1.0.0

 - Improved event ranking using a Resnet trained on XChem data
 - Improved hit detection using automated characterization set comparisons
 - Fragment autobuilding
 - Improved README.md with clearer instructions and reccomendations
 - Phenix dependency removed
 - PanDDA defaults to using present cifs with no rebuilding
 - Cleaner logs
 - Improved ranking for unbuildable events
 - Memory performance optimizations
 - Which filters to apply to datasets can be configured with `--data_quality_filters` and `--reference_comparability_filters`

## Planned for version 1.1.0

- Improved handling of DNA/RNA
- Support for cryoem maps
- Sites defined by residues instead of event centroids

## Reporting Errors

PanDDA 2 is still in development and feedback is always appreciated! 

If you have a problem with installation, program errors or uncertainty about the correctness of results the best place to get advice (and raise the issue to the developer) is at the XChem bullitin board: XCHEMBB@JISCMAIL.AC.UK
, which you can sign up for at https://www.jiscmail.ac.uk/cgi-bin/webadmin?A0=XCHEMBB.

If you have a problem with installation, then it is most likely specific to your system and best to just email me.

If the program errors while it runs, it is most helpful if you include the command line output and the json log in a GitHub issue. 

If you uncertain about the correctness of the results, then a GitHub issue is appropriate if you can share information publicly, in particular screenshots of maps or ligand fits. If you cannot, then an email to me is the best way to raise the concerns. Either way, please include the program output, the json log and screenshots of the offending z maps/event maps/autobuilds.






## Running PanDDA 2

PanDDA 2 supports the autobuilding of events and ranking them by autobuildability. All one needs to do is ensure that BUSTER is set up in their path (and hence ana_pdbmaps and rhofit).


Once you have installed PanDDA 2 in a conda environment, it can be run from that enviroment with autobuilding and automated ground state identification with the following:

1. Install ccp4/7.0.067 and PanDDA 2
2. Prepare your data for PanDDA 2
3. Run `pandda2.analyse` on your data to generate event maps
4. Run `pandda.inspect` to identify and complete fragment bound models
5. Decide which fragments to take forwards
6. Run `pandda.export` to prepare your results for refinement

## 1. Installation

First of all you will need to install `ccp4/7.0.067`. This is in order to access the functionality of `pandda.inspect` and `pandda.export`, which do not function correctly in more recent versions.

Once you have installed CCP4, it is recommended that you install PanDDA 2 in its own python 3.9 anaconda environment. If you do not have an Anaconda environment you can install one by following the instructions at https://www.anaconda.com/products/distribution#linux. 

Then:

```bash
git clone https://github.com/ConorFWild/pandda_2_gemmi.git
cd pandda_2_gemmi 
python -m pip install -e . 
python -m pip install ./_gemmi  # Install the custom version of gemmi
                                # This step will likely be removed in
                                # Future versions
```

The first time you run PanDDA 2 it will try to download two models of approximately 1GB total.

Installing PanDDA 2 this way will add various scripts to your path, but only while you are in this anaconda environment.

## 2. Preparing Data for PanDDA 2

The recommended pipeline with which to prepare PanDDA 2 input data is Dimple, which you can read more about at: https://ccp4.github.io/dimple/. The reccomended program for generating restraints is AceDRG (CCP4) or Grade (Global Phasing Ltd.).

The input directory for PanDDA must have the following format:

```commandline
<datasets directory>/<dataset name>/<dataset name>.pdb 
                                    <dataset name>.mtz
                                    compound/<compound name>.pdb
                                    compound/<compound name>.cif
```

Or, to give a concrete example:

```commandline
data_dirs
├── BAZ2BA-x434
    ├── BAZ2BA-x434.mtz
    ├── BAZ2BA-x434.pdb
    └── compound
        ├── ligand.cif
        └── ligand.pdb
 ...
```

### 3. Running PanDDA 2

The recommended way to run PanDDA 2 is:

```bash
python scripts/pandda.py --data_dirs=<data directories> --out_dir=<output directory> --pdb_regex=<pdb regex> --mtz_regex=<mtz regex> --local_cpus=<your number of cpus>
```

After PanDDA 2 has finished running, then results can be inspected with pandda.inspect as per PanDDA 1 (https://pandda.bitbucket.io/).

### 4. Inspecting the results of PanDDA 2
Refer to sections 8, 9 and 10 of the PanDDA tutorial. 

One important difference is that PanDDA 2 returns events ranked by score rather than by score per site. This means that 

### 5. Picking Fragment Hits

### 6. Exporting Results for Refinement

Refer to sections 11, 12 and 13 of the PanDDA tutorial.

## PanDDA 2 Usage FAQ

### PanDDA 2 has returned an enourmous number of events
This is working as intended: PanDDA 2 is designed re return significantly more events than PanDDA 1, but to rank them better. 

### The event map doesn't resemble the protein

This is to be expected if the event map's event does not is not a fragment or actual meaningful change: in such cases the event map is effectively random and should be ignored. An example of this is shown below:

![low_ranking_event](https://github.com/ConorFWild/pandda_2_gemmi/raw/0_1_0/imgs/low_ranking_event.png)


If there is clearly a fragment present but the event map's quality seems low for the protein: this is typically due to a poorly characterized ground state model and such event maps can be used as normal: i.e. can be built into. An example can be seen below:

### An event's density looks like the soaked fragment but it has not been autobuilt

There are two possibilities. The first is that autobuilding the event has failed, typically because the outlying Z-values form a small blob.

The second possibility is that the event has been autobuilt, but PanDDA prefers another event in the protein to merge into the model shown in pandda.inspect. This is because in the current version of PanDDA inspect only one autobuild can be chosen to display for any dataset, regarless of how many events it has. PanDDA does occasionally chose a poor autobuild over a good one, so it is important in such cases to manually build the good density and then check that if PanDDA has placed a fragment it too is in a sensible place.

### Many datasets don't seem to have any event maps

It is important to check the PanDDA logs to make sure that datasets you are interested in have not been filtered from consideration. 

### An event has been autobuilt but incorrectly

Autobuilding PanDDA events is not perfect and the method can sometimes fail. If you see an event in pandda.inspect that appears to resemble the fragment but the autobuild is poor, the advice is to correct the autobuild and save the new model.

## How PanDDA 2 works

PanDDA 2 differs from PanDDA 1 in two major ways: 

1. Firstly, it attempts to identify which sets of datasets should be used to produce a statistical model which has the optimal contrast (to each test dataset individually). This allows it to handle subtle heterogeneity between datasets. 

2. Secondly it attempts to autobuild the events returned, and then rank the events based on the quality of the model of the fragment that could be constructed. This allows improved rankings of events.

### Statistical Model Dataset Selection

PanDDA 2 selects the datasets to construct each statistical model it tries by identifying the sets of datasets which are closest to each other. This is achieved by:
 - Finding all pairwise distances between electron density maps
 - Finding the nearest neighbours of each dataset's electron density map
 - Finding the non-overlapping neighbourhoods with the minimum standard deviation between the maps they contain.

### Statistical Model Selection

Currently PanDDA 2 only chooses one statistical model to progress to event map generation and autobuilding. This is done by fitting several conformations of the expected fragment into each event from each statistical by differential evolution. 

The statistical which has the best fragment fit is then selected for progression to event map generation and autobuilding.

### Autobuilding

PanDDA 2 autobuilds the events of each dataset. This is done with a custom Rhofit invocation to account for the differing distribution of values between event maps and normal crystallographic 2Fo-Fc maps.

### Autobuild Selection

Current limitations with the interaction between pandda.inspect and PanDDA 2 mean that it is only possible to show one autobuild in the GUI. 

Therefore, the highest scoring autobuild by RSCC from any event in each dataset is selected and included in the initial model pandda.inspect shows the user. 

This has the effect that users may open apparently good hit density with no autobuild present, if another hit which is better fit by the autobuilding is present in the same dataset.

# 



