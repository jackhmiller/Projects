class TradeProfile:
    """Simple container for a single trade profile specification combining exposures and timing."""

    def __init__(self, exposures, offsets):
        """Creates a trade profile object.

        Parameters
        ----------
        exposures : tuple
            Tuple of target exposures, last element should always be zero, e.g. (1, -1, 0).
        offsets : tuple
            Tuple of offsets (trading days) from a liquidity event; < 0 for days prior to event,
            > 0 for days following event, 0 is the event date itself.

        Notes
        -----
        This is intended to be a very lightweight data container - no validation is done on the inputs!

        Examples
        --------
        Create a trade profile representing a short entry 3 days prior to event and exiting the following day after the event.

        >>> tp = TradeProfile((-1,0), (-3,1))
        >>> tp
        (-1, 0)_(-3, 1)

        Exposures and offsets can be accessed as properties of the object.

        >>> tp.exposures
        (-1, 1, 0)

        ...but not modified once instantiated, making it immutable (sort of); this will generate an exception:


        >>>tp.exposures = (1, 0)
        AttributeError                            Traceback (most recent call last)
        <ipython-input-42-da00d4303d77> in <module>
        ----> 1 tp.exposures = (1, 0)

        AttributeError: can't set attribute

        Instances of TradeProfile can be compared; identical exposures and offsets constitute equality.

        >>> tp == TradeProfile((-1, 0), (-3, 1))
        True

        TradeProfile instances are hashable, meaning they can be used as keys in a dictionary or entries in a Pandas
        index or as Pandas column labels.

        >>> x = {TradeProfile((-1, 0), (-3, 1)): 'a',
                 TradeProfile((1, 0), (-2, 0)): 'b'}
        >>> x[TradeProfile((1, 0), (-2, 0))]
        'b'
        -
        """

        self._exposures = exposures
        self._offsets = offsets[:len(exposures)]

    # Make these properties -> immutable
    @property
    def exposures(self):
        return self._exposures

    @property
    def offsets(self):
        return self._offsets

    def __repr__(self):
        return '{:}_{:}'.format(self.exposures, self.offsets)

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __hash__(self):
        return hash((self._exposures, self._offsets))

    def inverse(self):
        """Return a TradeProfile object with the exposures inverted.

        Examples
        --------
        >>> TradeProfile((-1, 0), (-3, 1)).inverse()
        (1, 0)_(-3, 1)
        """

        return TradeProfile(tuple(-y for y in self.exposures), self.offsets)

    def expand(self, n=3, null='-'):
        """Return tuple of (exposure 1, exposure 2, exposure 3, offset 1, offset 2, offset 3).

        Parameters
        ----------
        n : int; default 3
            Number of elements in expanded offsets, exposures collections; if length of offsets/exposures is less,
            pad with value specified by the 'null' parameter.
        null : default '-'
            Value with which to pad tuples out to n elements.

        Examples
        --------
        >>> tp = TradeProfile((-1, 0), (-3, 1))
        >>> tp.expand()
        (-1, 0, '-', -3, 1, '-')

        -
        """
        return tuple([*[*self.exposures, *([null] * n)][:n], *[*self.offsets, *([null] * n)][:n]])


def generate_trade_profiles(exposure_templates, max_offset):
    """Return a list of all possible TradeProfile variations combining exposure templates with days offset from event.

    Parameters
    ----------
    exposure_templates : list
        List of tuples representing +1 for long, -1 for short, 0 for flat e.g. [(-1, 0), (1, 0), ...].
    max_offset : int
        Maximum offset in days from the liquidity event date; all generated trade profiles will be bound to enter/exit
        at most at this maximum offset before/after the liquidity event date.

    Returns
    -------
    list

    Examples
    --------

    >>> generate_trade_profiles(exposure_templates=[(1, 0)], max_offset=2)
    [(1, 0)_(-2, -1),
     (1, 0)_(-2, 0),
     (1, 0)_(-2, 1),
     (1, 0)_(-2, 2),
     (1, 0)_(-1, 0),
     (1, 0)_(-1, 1),
     (1, 0)_(-1, 2),
     (1, 0)_(0, 1),
     (1, 0)_(0, 2),
     (1, 0)_(1, 2)]

    For example, the trade profile (1, 0)_(-2, 2) defines an agent that goes long 2 days prior to the
    liquidity event it trades, then goes flat 2 days after the event.

    Notes
    -----
    You can just work with one side of each possible pair (e.g. first exposure -> +1) then generate the inverse
    at a later stage (e.g. (1, 0) -> (-1, 0)) to improve computational efficiency.
    """

    trade_profiles = []

    for target in exposure_templates:
        for offset_1 in range(-max_offset, max_offset - (1 if len(target) == 3 else 0)):
            for offset_2 in range(offset_1 + 1, max_offset + (0 if len(target) == 3 else 1)):
                for offset_3 in range(offset_2 + 1, max_offset + 1) if len(target) == 3 else [None]:
                    trade_profiles += [TradeProfile(target, (offset_1, offset_2, offset_3))]

    return trade_profiles
