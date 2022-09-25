'''
Data loading and processing logic.
'''

import numpy as np
import random
import augs
import utils


def _read_image_robust(img_path, no_fail=False):
    '''
    Loads and returns an image that meets conditions along with a success flag, in order to avoid
    crashing.
    '''
    try:
        image = plt.imread(img_path).copy()
        success = True
        if (image.ndim != 3 or image.shape[2] != 3
                or np.any(np.array(image.strides) < 0)):
            # Either not RGB or has negative stride, so discard.
            success = False
            if no_fail:
                raise RuntimeError(f'ndim: {image.ndim}  '
                                   f'shape: {image.shape}  '
                                   f'strides: {image.strides}')

    except IOError as e:
        # Probably corrupt file.
        image = None
        success = False
        if no_fail:
            raise e

    return image, success


def _seed_worker(worker_id):
    '''
    Ensures that every data loader worker has a separate seed with respect to NumPy and Python
    function calls, not just within the torch framework. This is very important as it sidesteps
    lack of randomness- and augmentation-related bugs.
    '''
    worker_seed = torch.initial_seed() % (2 ** 32)  # This is distinct for every worker.
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_train_val_data_loaders(args, logger):
    '''
    return (train_loader, val_aug_loader, val_noaug_loader, dset_args).
    '''

    # TODO: Figure out noaug val dataset args as well.
    my_transform = augs.get_train_transform(args.image_dim)
    dset_args = dict()
    dset_args['transform'] = my_transform

    train_dataset = MyImageDataset(
        args.data_path, logger, 'train', **dset_args)
    val_aug_dataset = MyImageDataset(
        args.data_path, logger, 'val', **dset_args)
    val_noaug_dataset = MyImageDataset(
        args.data_path, logger, 'val', **dset_args) \
        if args.do_val_noaug else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)
    val_aug_loader = torch.utils.data.DataLoader(
        val_aug_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)
    val_noaug_loader = torch.utils.data.DataLoader(
        val_noaug_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False) \
        if args.do_val_noaug else None

    return (train_loader, val_aug_loader, val_noaug_loader, dset_args)


def create_test_data_loader(train_args, test_args, train_dset_args, logger):
    '''
    return (test_loader, test_dset_args).
    '''

    my_transform = augs.get_test_transform(train_args.image_dim)

    test_dset_args = copy.deepcopy(train_dset_args)
    test_dset_args['transform'] = my_transform

    test_dataset = MyImageDataset(
        test_args.data_path, logger, 'test', **test_dset_args)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_args.batch_size, num_workers=test_args.num_workers,
        shuffle=True, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)

    return (test_loader, test_dset_args)


class MyImageDataset(torch.utils.data.Dataset):
    '''
    PyTorch dataset class that returns uniformly random images of a labelled or unlabelled image
    dataset.
    '''

    def __init__(self, dataset_root, logger, phase, transform=None):
        '''
        :param dataset_root (str): Path to dataset (with or without phase).
        :param logger (MyLogger).
        :param phase (str): train / val_aug / val_noaug / test.
        :param transform: Data transform to apply on every image.
        '''
        # Get root and phase directories.
        phase_dir = os.path.join(dataset_root, phase)
        if not os.path.exists(phase_dir):
            # We may already be pointing to a phase directory (which, in the interest of
            # flexibility, is not necessarily the same as the passed phase argument).
            phase_dir = dataset_root
            dataset_root = str(pathlib.Path(dataset_root).parent)

        # Load all file paths beforehand.
        # NOTE: This method call handles subdirectories recursively, but also creates extra files.
        all_files = utils.cached_listdir(phase_dir, allow_exts=['jpg', 'jpeg', 'png'],
                                         recursive=True)
        file_count = len(all_files)
        print('Image file count:', file_count)
        dset_size = file_count

        self.dataset_root = dataset_root
        self.logger = logger
        self.phase = phase
        self.phase_dir = phase_dir
        self.transform = transform
        self.all_files = all_files
        self.file_count = file_count
        self.dset_size = dset_size

    def __len__(self):
        return self.dset_size

    def __getitem__(self, index):
        # TODO: Select either deterministic or random mode.
        # Sometimes, not every element in the dataset is actually suitable, in which case retries
        # may be needed, and as such the latter option is preferred.

        if 1:
            # Read the image at the specified index.
            file_idx = index
            image_fp = self.all_files[file_idx]
            rgb_input, _ = _read_image_robust(image_fp, no_fail=True)

        if 0:
            # Read a random image.
            success = True
            file_idx = -1
            while not success:
                file_idx = np.random.choice(self.file_count)
                image_fp = self.all_files[file_idx]
                rgb_input, success = _read_image_robust(image_fp)

        # Apply transforms.
        if self.transform is not None:
            rgb_input = self.src_transform(rgb_input)

        # Obtain ground truth.
        rgb_target = 1.0 - rgb_input

        # Return results.
        result = {'rgb_input': rgb_input,  # (H, W, 3).
                  'rgb_target': rgb_target,  # (H, W, 3).
                  'file_idx': file_idx,
                  'path': image_fp}
        return result
