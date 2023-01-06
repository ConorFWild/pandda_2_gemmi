from pandda_gemmi.analyse_interface import *
from pandda_gemmi.tables import (
    GetEventTable,
    GetSiteTable,
    SaveEvents
)

def get_site_table(pandda_args, console, pandda_fs_model, all_events, sites):
    # Output site table
    # with STDOUTManager('Building and outputting site table...', f'\tDone!'):
    console.start_site_table_output()
    # site_table: SiteTableInterface = SiteTable.from_events(all_events_sites,
    #                                                        pandda_args.max_site_distance_cutoff)
    site_table: SiteTableInterface = GetSiteTable()(all_events,
                                                    sites,
                                                    pandda_args.max_site_distance_cutoff)
    site_table.save(pandda_fs_model.analyses.pandda_analyse_sites_file)
    console.summarise_site_table_output(pandda_fs_model.analyses.pandda_analyse_sites_file)