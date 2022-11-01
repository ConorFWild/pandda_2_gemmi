def process_shell():
    homogenized_datasets  # Could be thread parallel, but not a huge issue

    aligned_maps  # clearly should be thread parallel

    models  #

    # Per dataset (a dataset, all model and a map need moving) - possibly could be python threaded as np and C++
    # release GIL?

    # # Per model (a dataset, a model, a map need moving)

    zmaps  # thread parallel, massive data moving, mostly C

    clusterings  # Possibly thread parallel, massive data moviing, mostly C

    events  # Very fast, mostly python

    # # # Per event (a dataset, two maps need moving)

    conformers  # Process parallel, low data moving, fast -> Probably best serial,

    # # # # Per conformer -> probably best serial

    optimize_score  # Probably only process parallel, massive data moving -> cheap, can be GIL released a decent amount

    rescore  # Process parallel, massive data moving -> cheap, can be gil released


    # # # # Per conf: cheap return

    # # # Per event: expensive return, a grid and bunch of complicated arrays

    # # Per model: Expensive return, a map and many nested complicated arrays (currently this is where multiprocessing is ^^")

    selected_model  # Not parallel

    output_maps()  # Process parallel, huge data moving, could probably be thread parallel

    # Per dataset: cheap return

