

def precission_recall_vs_fragalysis_from_pandda(pandda_dir):

    fragalysis_models = fragalysis.systems[SystemName.from_path(pandda_dir)].models

    pandda_dir_builds = PanDDADir.from_path(pandda_dir).models

    rmsds = [RMSD.from_structures(fragalysis_model, pandda_dir_build)
     for fragalysis_model, pandda_dir_build
     in [
         (fragalysis_model, pandda_dir_model)
        for
     ]]

    distplot(rmsds)



    #########

    fragalysis_models = fragalysis.systems[SystemName.from_path(pandda_dir)].models

    pandda_events = PanDDADir.from_path(pandda_dir).events

    matches = [model for event in pandda_events if distance(centroid(fragalysis_model), event.pos) < max_dist]
    unmatches = [
        model
        for model
        in fragalysis_models
        if min(
            [
                min(
                    [
                        distance(structure_centroid(ligand), event)
                        for event
                        in pandda_events
                        if event.dtag == model.dtag
                    ]
                )
                for ligand
                in ligands(model)
            ]
        )
    ]

