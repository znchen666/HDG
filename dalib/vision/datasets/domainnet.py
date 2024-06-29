import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class DomainNet(ImageList):

    # CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
    #            'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
    #            'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
    #            'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
    #            'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
    #            'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
    #            'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']

    def __init__(self, root, task, filter_class, split='all', **kwargs):
        if split == 'all':
            self.image_list = {
                "C": "image_list/Clipart_test.txt",
                "I": "image_list/Infograph_test.txt",
                "P": "image_list/Painting_test.txt",
                "Q": "image_list/Quickdraw_test.txt",
                "R": "image_list/Real_test.txt",
                "S": "image_list/Sketch_test.txt",
            }
        elif split == 'train':
            self.image_list = {
                "C": "image_list/Clipart_train.txt",
                "I": "image_list/Infograph_train.txt",
                "P": "image_list/Painting_train.txt",
                "Q": "image_list/Quickdraw_train.txt",
                "R": "image_list/Real_train.txt",
                "S": "image_list/Sketch_train.txt",
            }
        elif split == 'val':
            self.image_list = {
                "C": "image_list/Clipart_val.txt",
                "I": "image_list/Infograph_val.txt",
                "P": "image_list/Painting_val.txt",
                "Q": "image_list/Quickdraw_val.txt",
                "R": "image_list/Real_val.txt",
                "S": "image_list/Sketch_val.txt",
            }

        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        super(DomainNet, self).__init__(root, num_classes=len(filter_class), data_list_file=data_list_file,
                                       filter_class=filter_class, **kwargs)

        self.domain = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

    def __getitem__(self, index):
        """
        Parameters:
            - **index** (int): Index
            - **return** (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.data_list[index]
        domain_name = path.split('/')[7]
        domain_label = self.domain.index(domain_name)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target, domain_label

