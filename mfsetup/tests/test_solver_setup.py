from pathlib import Path

from mfsetup import MF6model, MFnwtModel
from mfsetup.fileio import dump


def test_solver_defaults(test_data_path, tmpdir):
    """Verify that default values aren't applied to solver
    packages if the simplified settings options are used
    (e.g. simple/moderate/complex)"""

    # modflow-6 IMS package
    mf6_model_config = test_data_path / 'pleasant_mf6_test.yml'
    cfg = MF6model.load_cfg(mf6_model_config)
    keep_keys = {'simulation', 'model',  'parent', 'setup_grid', 'dis', 'tdis',
                 'intermediate_data', 'postprocessing'}
    new_cfg = {k: v for k, v in cfg.items() if k in keep_keys}
    new_cfg['model']['packages'] = ['dis']
    new_cfg['ims'] = {'options': {'complexity': 'moderate'}}
    temp_yaml = Path(tmpdir) / 'junk.yml'
    dump(temp_yaml, new_cfg)
    m = MF6model.setup_from_yaml(temp_yaml)
    assert 'nonlinear' not in m.cfg['ims']
    assert 'linear' not in m.cfg['ims']

    # modflow-nwt NWT package
    mfnwt_model_config = test_data_path / 'pleasant_nwt_test.yml'
    cfg = MFnwtModel.load_cfg(mfnwt_model_config)
    keep_keys = {'simulation', 'model',  'parent', 'setup_grid', 'dis', 'bas6',
                 'intermediate_data', 'postprocessing'}
    new_cfg = {k: v for k, v in cfg.items() if k in keep_keys}
    new_cfg['model']['packages'] = ['dis', 'bas6']
    new_cfg['nwt'] = {'options': 'moderate'}
    temp_yaml = Path(tmpdir) / 'junk.yml'
    dump(temp_yaml, new_cfg)
    m = MFnwtModel.setup_from_yaml(temp_yaml)
    expected_keys = {'headtol', 'fluxtol', 'maxiterout', 'thickfact',
                     'linmeth', 'iprnwt', 'ibotav', 'Continue',
                     'use_existing_file', 'options'}
    assert not set(m.cfg['nwt'].keys()).difference(expected_keys)
    assert m.cfg['nwt']['options'] == 'moderate'
