from optyx.qpath import Split, Select, Id, Create, Merge
# from optyx.zw import Id

def test_qpath_zw_composition():
    num_op = Split() >> Select() @ Id(1) >> Create() @ Id(1) >> Merge()
