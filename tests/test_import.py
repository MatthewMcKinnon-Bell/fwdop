def test_import_fwdop():
    import importlib
    spec = importlib.util.find_spec('fwdop')
    assert spec is not None, "fwdop package not found on sys.path"
    import fwdop
    assert hasattr(fwdop, 'GFwdOp'), "GFwdOp not exported from fwdop"
