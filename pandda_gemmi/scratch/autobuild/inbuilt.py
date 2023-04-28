



class Rhofit:

    def __init__(self, cut=2.0):
        self.cut = cut

    def __call__(
            self,
            dmap_path,
            mtz_path,
            model_path,
            cif_path,
            out_dir,
    ):

        # Get the event grid from the dmap path


        # Mask protein
        inner_mask_grid = gemmi.Int8Grid(*grid.spacing)
        inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        inner_mask_grid.set_unit_cell(gemmi.UnitCell(*grid.unit_cell))
        for atom in reference.dataset.structure.protein_atoms():
            pos = atom.pos
            inner_mask_grid.set_points_around(pos,
                                              radius=1.25,
                                              value=1,
                                              )





        # Score
        time_scoring_start = time.time()
        results: Dict[Tuple[int, int], EventScoringResultInterface] = score_clusters(
            {(0, 0): event.cluster},
            {(0, 0): event_map_reference_grid},
            processed_dataset,
            res, rate, event_fit_num_trys,
            debug=debug,
        )
        time_scoring_finish = time.time()
        print(f"\t\t\tTime to actually score all events: {time_scoring_finish - time_scoring_start}")

        # Ouptut: actually will output only one result, so only one iteration guaranteed
        for result_id, result in results.items():
            # initial_score = result[0]
            # structure = result[1]
            initial_score = result.get_selected_structure_score()
            structure = result.get_selected_structure()

            for conformer_id, conformer_fitting_result in \
                    result.ligand_fitting_result.conformer_fitting_results.items():
                # if conformer_fitting_result:
                if conformer_fitting_result:
                    if conformer_fitting_result.score_log:
                        if "grid" in conformer_fitting_result.score_log:
                            del conformer_fitting_result.score_log["grid"]


        return AutobuildResult(
            log_result_dict,
            dmap_path,
            mtz_path,
            model_path,
            cif_path,
            out_dir
        )