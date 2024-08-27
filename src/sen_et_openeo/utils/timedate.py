import pandas as pd
import numpy as np


def find_closest_date(date, s2_dates):
    s2_dates = [pd.to_datetime(d) for d in s2_dates]
    date = pd.to_datetime(date)
    d = min(s2_dates, key=lambda x: abs(x - date))

    return d.strftime(format='%Y%m%d')


def _bracketing_dates(date_list, target_date):
    date_list = list(date_list)
    try:
        before = max([x for x in date_list if (
            target_date - x).total_seconds() >= 0])
        after = min([x for x in date_list if (
            target_date - x).total_seconds() <= 0])
    except ValueError:
        return None, None, np.nan
    if before == after:
        frac = 1
    else:
        frac = float((after - target_date).total_seconds()) / \
            float((after-before).total_seconds())
    return date_list.index(before), date_list.index(after), frac
