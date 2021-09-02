import pandas as pd
from tqdm.auto import tqdm


def generate_exposure_table(distance, trade_profiles):
    """Return a DataFrame of c.o.b. exposures for each agent specified by trade_profiles, given distance to event.

    Parameters
    ----------
    distance : time series
        Index is dates, values are days distance to the liquidity event; < 0 for distance to event in future,
        > 0 for distance from event in past.
    trade_profiles : list
        List of TradeProfile objects, specifying trade exposures and days offsets for each agent.

    Returns
    -------
    DataFrame : target exposures for each date (c.o.b.) in distance timeseries, for each exposure/offset combination
    specified in trade_profiles.

    Notes
    -----
    Column labels are the TradeProfile objects - these render as (exposures)_(offsets) when displayed; if string
    labels are required, simply
    convert to strings thus:

        df.set_axis(df.columns.map(str), axis=1, inplace=True)

    Examples
    --------
    >>> index = pd.date_range(freq='B', start='2008-03-26', periods=5)
    >>> data = [-3, -2, -1, 0, 1]
    >>> trade_profiles = [TradeProfile((1, 0), (-1, 1))]
    >>> distance = pd.Series(index=index, data=data)
    >>> generate_exposure_table(distance, trade_profiles)
                   (1, 0)_(-1, 1)
    2008-03-26                 0
    2008-03-27                 0
    2008-03-28                 1
    2008-03-31                 1
    2008-04-01                 0

    """

    df = pd.DataFrame(index=distance.index, columns=trade_profiles, dtype=object)
    df.loc[:, :] = 0

    for tp in trade_profiles:
        for exposure, offset_in, offset_out in zip(tp.exposures[:-1], tp.offsets, tp.offsets[1:]):
            df.loc[(distance.values >= offset_in) & (distance.values < offset_out), tp] = exposure

    return df


def add_inverse_exposure(exposure_tables):
    for event, et in exposure_tables.items():
        inverse_exposures = [x.inverse() for x in et]
        et_inverse = et.set_axis(inverse_exposures, axis=1, inplace=False)
        et = et.merge(et_inverse, left_index=True, right_index=True)

    return et


def all_exposure_tables(trade_profiles, df_event_distance):
    """Generate the full space of all possible agents"""
    exposure_tables = {event: generate_exposure_table(distance, trade_profiles) for event, distance in
                       tqdm(list(df_event_distance.items()), desc='Generating exposure tables')}

    if len(list(df_event_distance.items())) == 1:
        exp = exposure_tables['Distance']
        inverse_exposures = [x.inverse() for x in exp]
        et_inverse = exp.set_axis(inverse_exposures, axis=1, inplace=False)
        et = exp.merge(-et_inverse, left_index=True, right_index=True)
    else:
        et = add_inverse_exposure(exposure_tables)

    return et


###########
def combine_exposure_tables(packages_by_event, exposure_tables, add_inverse=True):
    """Merge exposure table DataFrames in a dictionary into a single DataFrame.

    Parameters
    ----------
    packages_by_event : dict
        Dict of packages to be traded for each event code e.g. {'A10Y': ['Bond10Y', 'Curve7Y10Y', 'Fly7Y10Y30Y']}.
    exposure_tables : dict
        Dictionary of exposure DataFrames for each event in event_codes, providing historical c.o.b. exposures for
        all trade profiles.
    add_inverse : bool, default False
        If True, augments the output with the inverse trading profiles, e.g. TradeProfile((-1, 0), (-2, 1))
        also generates P&L for TradeProfile((1, 0), (-2, 1))

    Returns
    -------
    DataFrame : Exposure profiles of all packages, column labels are MultiIndex of (event, package, trade profile).
    """

    combined = None

    for event, et in exposure_tables.items():

        if add_inverse:
            inverse_exposures = [x.inverse() for x in et]
            et_inverse = et.set_axis(inverse_exposures, axis=1, inplace=False)
            et = et.merge(et_inverse, left_index=True, right_index=True)

        for package in packages_by_event[event]:
            df = et.set_axis(pd.MultiIndex.from_product([[event], [package], et.columns]), axis=1, inplace=False)
            combined = combined.merge(df, left_index=True, right_index=True) if combined is not None else df

        combined.columns.names = ['Event', 'Package', 'Profile']

    return combined
