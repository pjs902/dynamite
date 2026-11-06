#!/usr/bin/env python3
"""parameter_space_settings: reuse_orblib_across_ml.

Default (True): models differing only in ml share one orbit library and are
velocity-rescaled from its ml. False: ml identifies the library too, so each
of those models gets its own library at its own ml (scaling factor 1).

Run from dev_tests/:  python test_orblib_ml_reuse.py
"""
import os

import numpy as np

import dynamite as dyn
from dynamite.model_iterator import ModelInnerIterator


def _config(reuse):
    c = dyn.config_reader.Configuration('user_test_config_ml.yaml',
                                        reset_logging=False)
    if reuse is not None:
        c.settings.parameter_space_settings['reuse_orblib_across_ml'] = reuse
    t = c.all_models.table
    t.remove_rows(slice(None))
    p0 = {n: p.par_value for n, p in zip(c.parspace.par_names, c.parspace)}
    ml_q = [(p0['ml'], 0), (p0['ml'] + 1., 0), (p0['ml'], 1)]  # 2nd: only ml differs
    qname = next(n for n in c.parspace.par_names if n.startswith('q-'))
    for ml, dq in ml_q:
        vals = dict(p0, ml=ml, **{qname: p0[qname] - 0.01 * dq})
        n_par = len(c.parspace.par_names)
        row = [vals[n] for n in c.parspace.par_names]
        for col in t.colnames[n_par:]:
            row.append({'time_modified': '', 'which_iter': 0, 'directory': ''}
                       .get(col, t[col].dtype.type(0)))
        t.add_row(row)
    return c


def test_default_shares_orblib_across_ml():
    c = _config(None)
    it = ModelInnerIterator(config=c, par_generator=None)
    assert 'ml' not in it.orblib_parameters
    assert [it.is_new_orblib(i) for i in range(3)] == [True, False, True]
    assert c.all_models.get_ml_of_original_orblib(1) == c.all_models.table['ml'][0]
    it.assign_model_directories(rows_orblib=[0, 2], rows_ml=[1])
    d = c.all_models.table['directory']
    assert d[0].split('/')[0] == d[1].split('/')[0] != d[2].split('/')[0], list(d)
    print('  default_shares_orblib_across_ml OK')


def test_no_reuse_gives_each_ml_its_own_orblib():
    c = _config(False)
    it = ModelInnerIterator(config=c, par_generator=None)
    assert 'ml' in it.orblib_parameters
    assert [it.is_new_orblib(i) for i in range(3)] == [True, True, True]
    for i in range(3):
        assert c.all_models.get_ml_of_original_orblib(i) == c.all_models.table['ml'][i]
    it.assign_model_directories(rows_orblib=[0, 1, 2])
    tops = [d.split('/')[0] for d in c.all_models.table['directory']]
    assert len(set(tops)) == 3, tops
    print('  no_reuse_gives_each_ml_its_own_orblib OK')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_default_shares_orblib_across_ml()
    test_no_reuse_gives_each_ml_its_own_orblib()
    print('ALL PASSED')
