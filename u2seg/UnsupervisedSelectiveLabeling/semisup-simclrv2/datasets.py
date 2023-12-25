from torchvision.datasets.folder import ImageFolder, default_loader

# Credit: https://github.com/amazon-research/exponential-moving-average-normalization/blob/main/data/datasets.py
class ImageFolderWithIndex(ImageFolder):

    def __init__(self, root, indexs=None, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)
        if indexs is not None:
            self.samples = [self.samples[i] for i in indexs]
            self.targets = [self.targets[i] for i in indexs]
            self.imgs = self.samples

class ImageFolderWithIndexAndTarget(ImageFolder):

    def __init__(self, root, indexs=None, transform=None, target_transform=None, target=None,
                 loader=default_loader, is_valid_file=None):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)
        assert target is not None, "need to provide targets"
        if indexs is not None:
            assert len(indexs) == len(target), "{} != {}".format(len(indexs), len(target))
            
            self.targets = target

            # Use the provided target instead of the ImageFolder target
            self.samples = [(self.samples[i][0], target_item) for i, target_item in zip(indexs, target)]
            self.imgs = self.samples
