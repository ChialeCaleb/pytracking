from collections import namedtuple
import importlib
from pytracking.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "pytracking.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    eotb=DatasetInfo(module=pt % "eotb", class_name="EOTBDataset", kwargs=dict()),
)


def load_dataset(name: str,splits = None):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    sequence_list = dataset.get_sequence_list()
    if splits is not None:
        train_list = [f.strip() for f in open('../eotb_'+splits+'_split.txt', 'r').readlines()]
        sequence_list = [i for i in dataset.get_sequence_list() if i.name in train_list]
    return sequence_list


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset

def get_traindataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name,'train'))
    return dset

def get_valdataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name,'val'))
    return dset