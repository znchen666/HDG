import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class PACS(ImageList):


    # CLASSES = [dog  elephant  giraffe  guitar  horse  house  person]

    def __init__(self, root, task, filter_class, split='all', **kwargs):
        if split == 'all':
            self.image_list = {
                "A": "image_list/art_painting_test_kfold.txt",
                "C": "image_list/cartoon_test_kfold.txt",
                "P": "image_list/photo_test_kfold.txt",
                "S": "image_list/sketch_test_kfold.txt",
            }
        elif split == 'train':
            self.image_list = {
                "A": "image_list/art_painting_train_kfold.txt",
                "C": "image_list/cartoon_train_kfold.txt",
                "P": "image_list/photo_train_kfold.txt",
                "S": "image_list/sketch_train_kfold.txt",
            }
        elif split == 'val':
            self.image_list = {
                "A": "image_list/art_painting_crossval_kfold.txt",
                "C": "image_list/cartoon_crossval_kfold.txt",
                "P": "image_list/photo_crossval_kfold.txt",
                "S": "image_list/sketch_crossval_kfold.txt",
            }

        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])
        super(PACS, self).__init__(root, num_classes=len(filter_class), data_list_file=data_list_file,
                                       filter_class=filter_class, **kwargs)
        self.domain = ["art_painting", "cartoon", "photo", "sketch"]

    def __getitem__(self, index):
        """
        Parameters:
            - **index** (int): Index
            - **return** (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.data_list[index]
        domain_name = path.split('/')[8]
        domain_label = self.domain.index(domain_name)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target, domain_label


