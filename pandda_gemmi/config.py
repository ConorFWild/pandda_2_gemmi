import typing
import dataclasses

import argparse
from pathlib import Path

from pandda_gemmi.types import *


@dataclasses.dataclass()
class DatasetFlags:
    ground_state_datasets: typing.List[Dtag] = None
    exclude_from_z_map_analysis: typing.List[Dtag] = None
    exclude_from_characterisation: typing.List[Dtag] = None
    only_datasets: typing.List[Dtag] = None
    ignore_datasets: typing.List[Dtag] = None

    @classmethod
    def from_args(cls, args):
        return DatasetFlags(ground_state_datasets=[Dtag(dtag) for dtag in args.ground_state_datasets.split(",")],
                            exclude_from_z_map_analysis=[Dtag(dtag) for dtag
                                                         in args.exclude_from_z_map_analysis.split(",")],
                            exclude_from_characterisation=[Dtag(dtag) for dtag
                                                           in args.exclude_from_characterisation.split(",")],
                            only_datasets=[Dtag(dtag) for dtag in args.only_datasets.split(",")],
                            ignore_datasets=[Dtag(dtag) for dtag in args.ignore_datasets.split(",")],
                            )


@dataclasses.dataclass()
class Input:
    data_dirs: Path
    mtz_regex: str
    pdb_regex: str
    dataset_flags: DatasetFlags

    @classmethod
    def from_args(cls, args):
        return Input(data_dirs=Path(args.data_dirs),
                     mtz_regex=args.mtz_regex,
                     pdb_regex=args.pdb_regex,
                     dataset_flags=DatasetFlags.from_args(args),
                     )


@dataclasses.dataclass()
class ResolutionBinning:
    dynamic_res_limits: bool = True
    high_res_upper_limit: float = 0.0
    high_res_lower_limit: float = 4.0
    high_res_increment: float = 0.05
    max_shell_datasets: int = 60
    min_characterisation_datasets: int = 60

    @classmethod
    def from_args(cls, args):
        return ResolutionBinning(dynamic_res_limits=args.dynamic_res_limits,
                                 high_res_upper_limit=args.high_res_upper_limit,
                                 high_res_lower_limit=args.high_res_lower_limit,
                                 high_res_increment=args.high_res_increment,
                                 max_shell_datasets=args.max_shell_datasets,
                                 min_characterisation_datasets=args.min_characterisation_datasets
                                 )


@dataclasses.dataclass()
class DiffractionData:
    structure_factors: StructureFactors = StructureFactors.from_string("FWT,PHWT")
    low_resolution_completeness: float = 4.0
    all_data_are_valid_values: bool = True

    @classmethod
    def from_args(cls, args):
        return DiffractionData(structure_factors=StructureFactors.from_string(args.structure_factors),
                               low_resolution_completeness=args.low_resolution_completeness,
                               all_data_are_valid_values=args.all_data_are_valid_values,
                               )


@dataclasses.dataclass()
class Filtering:
    max_rfree: float = 0.4
    max_rmsd_to_reference: float = 1.5
    max_wilson_plot_z_score: float = 5.0
    same_space_group_only: bool = True
    similar_models_only: bool = False

    @classmethod
    def from_args(cls, args):
        return Filtering(max_rfree=args.max_rfree,
                         max_rmsd_to_reference=args.max_rmsd_to_reference,
                         max_wilson_plot_z_score=args.max_wilson_plot_z_score,
                         same_space_group_only=args.same_space_group_only,
                         similar_models_only=args.similar_models_only,
                         )


@dataclasses.dataclass()
class BlobFinding:
    min_blob_volume: float = 10.0
    min_blob_z_peak: float = 3.0
    clustering_cutoff: float = 1.732

    @classmethod
    def from_args(cls, args):
        return BlobFinding(min_blob_volume=args.min_blob_volume,
                           min_blob_z_peak=args.min_blob_z_peak,
                           clustering_cutoff=args.clustering_cutoff,
                           )


@dataclasses.dataclass()
class BackgroundCorrection:
    min_bdc: float = 0.0
    max_bdc: float = 1.0
    increment: float = 0.01
    output_multiplier: float = 1.0

    @classmethod
    def from_args(cls, args):
        return BackgroundCorrection(min_bdc=args.min_bdc,
                                    max_bdc=args.max_bdc,
                                    increment=args.increment,
                                    output_multiplier=args.output_multiplier,
                                    )


@dataclasses.dataclass()
class Processing:
    process_dict_n_cpus: int = 12
    process_shells: str = "luigi"
    h_vmem: float = 100
    m_mem_free: float = 5

    @classmethod
    def from_args(cls, args):
        return Processing(process_dict_n_cpus=args.process_dict_n_cpus,
                          process_shells=args.process_shells,
                          h_vmem=args.h_vmem,
                          m_mem_free=args.m_mem_free,
                          )


@dataclasses.dataclass()
class MapProcessing:
    resolution_factor: float = 0.25
    grid_spacing: float = 0.5
    padding: float = 3
    density_scaling: str = "sigma"

    @classmethod
    def from_args(cls, args):
        return MapProcessing(resolution_factor=args.resolution_factor,
                             grid_spacing=args.grid_spacing,
                             padding=args.padding,
                             density_scaling=args.density_scaling,
                             )


@dataclasses.dataclass()
class Masks:
    outer_mask: float = 6
    inner_mask: float = 1.8
    inner_mask_symmetry: float = 3.0
    contour_level: float = 2.5
    negative_values: bool = False

    @classmethod
    def from_args(cls, args):
        return Masks(outer_mask=args.outer_mask,
                     inner_mask=args.inner_mask,
                     inner_mask_symmetry=args.inner_mask_symmetry,
                     contour_level=args.contour_level,
                     negative_values=args.negative_values,
                     )


@dataclasses.dataclass()
class Params:
    resolution_binning: ResolutionBinning
    diffraction_data: DiffractionData
    filtering: Filtering
    map_processing: MapProcessing
    masks: Masks
    blob_finding: BlobFinding
    background_correction: BackgroundCorrection
    processing: Processing

    @classmethod
    def from_args(cls, args):
        return cls(resolution_binning=ResolutionBinning.from_args(args),
                   diffraction_data=DiffractionData.from_args(args),
                   filtering=Filtering.from_args(args),
                   masks=Masks.from_args(args),
                   map_processing=MapProcessing.from_args(args),
                   blob_finding=BlobFinding.from_args(args),
                   background_correction=BackgroundCorrection.from_args(args),
                   processing=Processing.from_args(args),
                   )


@dataclasses.dataclass()
class Output:
    out_dir: Path

    @classmethod
    def from_args(cls, args):
        return cls(out_dir=args.out_dir)


@dataclasses.dataclass()
class Config:
    input: Input
    output: Output
    params: Params

    @staticmethod
    def from_args():
        parser = argparse.ArgumentParser()

        # Input
        parser.add_argument("--data_dirs",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--pdb_regex",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--mtz_regex",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        # Dataset selection
        parser.add_argument("--ground_state_datasets",
                            default="",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--exclude_from_z_map_analysis",
                            default="",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--exclude_from_characterisation",
                            default="",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--only_datasets",
                            default="",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--ignore_datasets",
                            default="",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        # Output
        parser.add_argument("--out_dir",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        # params
        # Res limits
        parser.add_argument("--dynamic_res_limits",
                            default=True,
                            type=bool,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--high_res_upper_limit",
                            default=0.0,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--high_res_lower_limit",
                            default=4.0,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--high_res_increment",
                            default=0.05,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--max_shell_datasets",
                            default=60,
                            type=int,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--min_characterisation_datasets",
                            default=60,
                            type=int,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )


        # Diffraction data
        parser.add_argument("--structure_factors",
                            default="FWT,PHWT",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--all_data_are_valid_values",
                            default=True,
                            type=bool,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--low_resolution_completeness",
                            default=4.0,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        # Filters
        parser.add_argument("--max_rmsd_to_reference",
                            default=1.5,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--max_rfree",
                            default=0.4,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        # Maps
        parser.add_argument("--resolution_factor",
                            default=0.25,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--grid_spacing",
                            default=0.5,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--padding",
                            default=3,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--density_scaling",
                            default="sigma",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        # masks
        parser.add_argument("--outer_mask",
                            default=1.8,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--inner_mask",
                            default=1.8,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--inner_mask_symmetry",
                            default=3.0,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--contour_level",
                            default=2.5,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--negative_values",
                            default=False,
                            type=bool,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        # blob finding
        parser.add_argument("--min_blob_volume",
                            default=10.0,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--min_blob_z_peak",
                            default=3.0,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--clustering_cutoff",
                            default=1.732,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        # background correction
        parser.add_argument("--min_bdc",
                            default=0.0,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--max_bdc",
                            default=1.0,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--increment",
                            default=0.01,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--output_multiplier",
                            default=1.0,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        # processing
        parser.add_argument("--process_dict_n_cpus",
                            default=12,
                            type=int,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--process_shells",
                            default="luigi",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--h_vmem",
                            default=100,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )
        parser.add_argument("--m_mem_free",
                            default=5,
                            type=float,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        args = parser.parse_args()

        input: Input = Input.from_args(args)
        output: Output = Output.from_args(args)
        params: Params = Params.from_args(args)

        return Config(input=input,
                      output=output,
                      params=params,
                      )
