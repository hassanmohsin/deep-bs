import torch.utils.data


class BaseDataLoader():
    def __init__(self): pass

    def load_data(): return None

    def initialize(self, opt): self.opt = opt


def CreateDataset(opt):
    '''
    opt.csvfile
    opt.dataroot
    opt.filter_kd
    '''
    dataset = None
    if opt.dataset_mode == 'pdbbind':
        from .pdbbind_dataset import PdbBindDataset
        dataset = PdbBindDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=opt.batch_size,
            shuffle=not opt.serial_batches, num_workers=int(opt.nThreads),
            pin_memory=True)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size: break
            yield data

    def name(self):
        return 'CustomDatasetDataLoader'

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
