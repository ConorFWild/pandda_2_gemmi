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

Once you have installed PanDDA 2 in a conda environment, it can be run from that enviroment with autobuilding and automated ground state identification with the following:

1. A. Install ccp4/7.0.067 and PanDDA 2 OR B. Install PanDDA 2 and PanDDA 2 Inspect (beta)
2. Prepare your data for PanDDA 2
3. Run `pandda2.analyse` on your data to generate event maps
4. Run `pandda.inspect` to identify and complete fragment bound models
5. Decide which fragments to take forwards
6. Run `pandda.export` to prepare your results for refinement

## 1. Installation

Once you have installed CCP4, it is recommended that you install PanDDA 2 in its own python 3.9 anaconda environment. If you do not have an Anaconda environment you can install one by following the instructions at https://www.anaconda.com/products/distribution#linux. 

With Anaconda an environment can be setup like so:

```bash
conda create -n pandda2 python=3.9
conda install llvm python-devtools # If building gemmi from source
```

Then:

```bash
git clone https://github.com/ConorFWild/pandda_2_gemmi.git
cd pandda_2_gemmi 
python -m pip install -e . 

```

The first time you run PanDDA 2 it will try to download two models of approximately 1GB total.

Installing PanDDA 2 this way will add various scripts to your path, but only while you are in this anaconda environment.

Then:

### 1.A. Install CCP4

You will need to install `ccp4/7.0.067`. This is in order to access the functionality of `pandda.inspect` and `pandda.export`, which do not function correctly in more recent versions.

### 1.B. Install PanDDA 2 Inspect (beta)

There is now a modern Moorhen based `pandda.inspect`, which can be installed by following the instructions at:
https://github.com/ConorFWild/PanDDA2Inspect

## 2. Preparing Data for PanDDA 2

The recommended pipeline with which to prepare PanDDA 2 input data is Dimple, which you can read more about at: https://ccp4.github.io/dimple/. The reccomended program for generating restraints is Grade (Global Phasing Ltd.), or AceDRG (CCP4) if you do not have access to Grade.

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
pandda2.analyse --data_dirs=<data directories> --out_dir=<output directory> --pdb_regex=<pdb regex> --mtz_regex=<mtz regex> --local_cpus=<your number of cpus>
```

OR

```bash
python scripts/pandda.py --data_dirs=<data directories> --out_dir=<output directory> --pdb_regex=<pdb regex> --mtz_regex=<mtz regex> --local_cpus=<your number of cpus>
```

After PanDDA 2 has finished running, then results can be inspected with pandda.inspect as per PanDDA 1 (https://pandda.bitbucket.io/).

### 4. Inspecting the results of PanDDA 2
Refer to sections 8, 9 and 10 of the PanDDA tutorial. 

One important difference is that PanDDA 2 returns events ranked by score rather than by score per site. This means that 

### 5. Exporting Results for Refinement

Refer to sections 11, 12 and 13 of the PanDDA tutorial.
 

