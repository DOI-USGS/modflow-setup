import pandas as pd


def setup_wel_data(model):
    """Performs the part of well package setup that is independent of
    MODFLOW version. Returns a DataFrame with the information
    needed to set up stress_period_data.
    """
    # master dataframe for stress period data
    df = pd.DataFrame(columns=['per', 'k', 'i', 'j', 'flux', 'comments'])

    # multiple types of source data can be submitted
    datasets = model.cfg['source_data'].keys()
    for k, v in datasets:

        # determine the format
        if k.lower() == 'csv':  # generic csv
            raise NotImplemented('Generic csv format')
            aw = pd.read_csv(added_wells)
            aw.rename(columns={'name': 'comments'}, inplace=True)
        elif k.lower() == 'wells':  # generic dict
            added_wells = {k: v for k, v in added_wells.items() if v is not None}
            if len(added_wells) > 0:
                aw = pd.DataFrame(added_wells).T
                aw['comments'] = aw.index
            else:
                aw = None

        elif k.lower() == 'wdnr_dataset':

            from .wateruse import get_mean_pumping_rates, resample_pumping_rates

            # Get steady-state pumping rates
            check_source_files([v['water_use'],
                                v['water_use_points']])

            # fill out period stats
            period_stats = v['period_stats']
            if isinstance(period_stats, str):
                period_stats = {kper: period_stats for kper in range(model.nper)}

            # separate out stress periods with period mean statistics vs.
            # those to be resampled based on start/end dates
            resampled_periods = {k: v for k, v in period_stats.items()
                                 if v == 'resample'}
            periods_with_dataset_means = {k: v for k, v in period_stats.items()
                                          if k not in resampled_periods}

            if len(periods_with_dataset_means) > 0:
                wu_means = get_mean_pumping_rates(model.cfg['source_data']['water_use'],
                                                  model.cfg['source_data']['water_use_points'],
                                                  period_stats=periods_with_dataset_means,
                                                  model=model)
                df = df.append(wu_means)
            if len(resampled_periods) > 0:
                wu_resampled = resample_pumping_rates(model.cfg['source_data']['water_use'],
                                                      model.cfg['source_data']['water_use_points'],
                                                      model=model)
                df = df.append(wu_resampled)

    # boundary fluxes from parent model
    if model.perimeter_bc_type == 'flux':
        assert model.parent is not None, "need parent model for TMR cut"

        # boundary fluxes
        kstpkper = [(0, 0)]
        tmr = Tmr(model.parent, model)

        # parent periods to copy over
        kstpkper = [(0, per) for per in model.cfg['model']['parent_stress_periods']]
        bfluxes = tmr.get_inset_boundary_fluxes(kstpkper=kstpkper)
        bfluxes['comments'] = 'boundary_flux'
        df = df.append(bfluxes)

    # added wells
    added_wells = model.cfg['wel'].get('added_wells')
    if added_wells is not None:
        if isinstance(added_wells, str):
            aw = pd.read_csv(added_wells)
            aw.rename(columns={'name': 'comments'}, inplace=True)
        elif isinstance(added_wells, dict):
            added_wells = {k: v for k, v in added_wells.items() if v is not None}
            if len(added_wells) > 0:
                aw = pd.DataFrame(added_wells).T
                aw['comments'] = aw.index
            else:
                aw = None
        elif isinstance(added_wells, pd.DataFrame):
            aw = added_wells
            aw['comments'] = aw.index
        else:
            raise IOError('unrecognized added_wells input')

        if aw is not None:
            if 'x' in aw.columns and 'y' in aw.columns:
                aw['i'], aw['j'] = model.modelgrid.intersect(aw['x'].values,
                                                            aw['y'].values)

            aw['per'] = aw['per'].astype(int)
            aw['k'] = aw['k'].astype(int)
            df = df.append(aw)

    df['per'] = df['per'].astype(int)
    df['k'] = df['k'].astype(int)
    # Copy fluxes to subsequent stress periods as necessary
    # so that fluxes aren't unintentionally shut off;
    # for example if water use is only specified for period 0,
    # but the added well pumps in period 1, copy water use
    # fluxes to period 1.
    last_specified_per = int(df.per.max())
    copied_fluxes = [df]
    for i in range(last_specified_per):
        # only copied fluxes of a given stress period once
        # then evaluate copied fluxes (with new stress periods) and copy those once
        # after specified per-1, all non-zero fluxes should be propegated
        # to last stress period
        # copy non-zero fluxes that are not already in subsequent stress periods
        if i < len(copied_fluxes):
            in_subsequent_periods = copied_fluxes[i].comments.duplicated(keep=False)
            # (copied_fluxes[i].per < last_specified_per) & \
            tocopy = (copied_fluxes[i].flux != 0) & \
                     ~in_subsequent_periods
            if np.any(tocopy):
                copied = copied_fluxes[i].loc[tocopy].copy()

                # make sure that wells to be copied aren't in subsequent stress periods
                duplicated = np.array([r.comments in df.loc[df.per > i, 'comments']
                                       for idx, r in copied.iterrows()])
                copied = copied.loc[~duplicated]
                copied['per'] += 1
                copied_fluxes.append(copied)
    df = pd.concat(copied_fluxes, axis=0)
    wel_lookup_file = os.path.join(model.model_ws, os.path.split(model.cfg['wel']['lookup_file'])[1])
    model.cfg['wel']['lookup_file'] = wel_lookup_file

    # save a lookup file with well site numbers/categories
    df[['per', 'k', 'i', 'j', 'flux', 'comments']].to_csv(wel_lookup_file, index=False)
    return df