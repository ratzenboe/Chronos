import numpy as np


def chunks(l, n):
    """Yield successive n-sized chunks from l.
    Split the processes up into n and then chunk the lines
    """
    for i in range(0, len(l), n):
        yield l[i:i + n], range(i, i + len(l[i:i + n]))


def resolve_duplicates(x):
    unique, counts = np.unique(x, return_counts=True)
    dup = unique[counts > 1]
    # Add small random effect
    x += np.isin(x, dup).astype(np.float64)*np.random.normal(0, 1e-5, x.size)
    return x


def isin_range(arr, *args):
    lo = np.min(args)
    hi = np.max(args)
    return (arr >= lo) & (arr <= hi)


def merge_dicts(*dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result
