import os
from enum import Enum

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# Image_MEAN = [0.3957,] #　only for class 2
# Image_STD = [0.1276,]


class DAGMDataset(Dataset):

    def __init__(
            self,
            source,
            classname,
            mean=0.5,
            std=0.5,
            resize=256,
            imagesize=224,
            split='train',
            **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classname = classname

        self.data_to_iterate = self.get_image_data()

        self.transform_img = transforms.Compose([transforms.Resize(resize),
                                                 transforms.RandomCrop(imagesize),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=mean, std=std)
                                                 ] if split == 'train'
                                                else [transforms.Resize(resize),
                                                      transforms.CenterCrop(imagesize),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=mean, std=std)
                                                      ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])
        self.imagesize = (1, imagesize, imagesize)
        self.transform_mean = mean
        self.transform_std = std

    def __getitem__(self, idx):
        label, image_path, mask_path, name = self.data_to_iterate[idx]
        image = Image.open(image_path).convert("L")
        image = self.transform_img(image)

        if self.split == 'test' and mask_path is not None:
            mask = Image.open(mask_path).convert('L')
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros(self.imagesize)

        return {
            "image": image,
            'name': name,
            "mask": mask,
            "label": label,
            "classname": self.classname
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        data_to_iterate = None
        if self.split in ('train', 'val'):
            classpath = os.path.join(self.source, self.classname, self.split)
            imgs = os.listdir(classpath)
            imgs = [os.path.join(classpath, x) for x in imgs if x.endswith('.PNG')]
            data_to_iterate = [[None for _ in range(4)] for _ in
                               range(len(imgs))]  # 四元组为（label, imgpath, maskpath, name）
            for _ in range(len(imgs)):
                data_to_iterate[_][0] = 0
                data_to_iterate[_][1] = imgs[_]
                data_to_iterate[_][3] = os.path.split(imgs[_])[-1]
        elif self.split == 'test':
            classpath = os.path.join(self.source, self.classname, self.split)
            types = os.listdir(classpath)
            rec = {}
            for typ in types:
                if typ != 'mask':
                    imgpaths = os.listdir(os.path.join(classpath, typ))
                    imgs = [os.path.join(classpath, typ, x) for x in imgpaths if x.endswith('.PNG')]

                    rec[typ] = [[None for _ in range(4)] for _ in range(len(imgs))]
                    for _ in range(len(imgs)):
                        img_name = os.path.split(imgs[_])[-1]
                        rec[typ][_][0] = 0
                        rec[typ][_][1] = imgs[_]
                        rec[typ][_][3] = img_name
                        if typ == 'defect':
                            rec[typ][_][0] = 1
                            mask_name = img_name[:4] + '_label.PNG'
                            mask_path = os.path.join(classpath, 'mask', mask_name)
                            rec[typ][_][2] = mask_path
            data_to_iterate = []
            for key in rec.keys():
                data_to_iterate.extend(rec[key])

        else:
            raise KeyError(f"{self.split} not in ('train', 'val', 'test')")

        return data_to_iterate


class TestDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, resize: int = 256, imagesize: int = 224,
                 mask_suffix: str = '_label.PNG', classname: str = 'class6'):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.classname = classname
        self.imagesize = imagesize
        self.imgs = [os.path.split(file)[-1] for file in os.listdir(images_dir) if file.endswith('.PNG')]
        labels = []
        masks = []
        for file in self.imgs:
            mask_path = os.path.join(mask_dir, file[:4] + mask_suffix)
            if os.path.exists(mask_path):
                labels.append(1)
                masks.append(mask_path)
            else:
                labels.append(0)
                masks.append(None)
        # assert len(self.imgs_A) != 0 ,'数据集A为空'
        # assert len(self.imgs_B) != 0 ,'数据集B为空'
        print(f'Creating datasetA with {len(self.imgs) - sum(labels)} examples and datasetsB with {sum(labels)}')
        self.labels = labels
        self.masks = masks

        self.transform_img = transforms.Compose([transforms.Resize(resize),
                                                 transforms.CenterCrop(imagesize),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=0.5, std=0.5)
                                                 ])
        self.transform_mask = transforms.Compose([transforms.Resize(resize),
                                                  transforms.CenterCrop(imagesize),
                                                  transforms.ToTensor(),
                                                  ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        name = self.imgs[idx]
        img = Image.open(os.path.join(self.images_dir, self.imgs[idx])).convert('L')
        img = self.transform_img(img)
        mask_path = self.masks[idx]
        if mask_path:
            mask = Image.open(mask_path).convert('L')
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros((1, self.imagesize, self.imagesize))

        label = self.labels[idx]

        return {
            'image': img,
            'mask': mask,
            'label': label,
            'name': name
        }


class MVTecDataset(Dataset):
    _CLASSNAMES = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    def __init__(self, source='../../../mvtec_anomaly_detection', resize=256, imagesize=224, split='train', gray=False,**kwargs):
        super().__init__()
        self.source = source
        self.split = split
        self.gray = gray
        self.class2label = {item: i for i, item in enumerate(self._CLASSNAMES)}
        self.label2class = {i: item for item, i in self.class2label.items()}

        self.transform_mean = [0.5]
        self.transform_std = [0.5]
        self.transform_img = transforms.Compose([transforms.Resize(resize),
                                                 transforms.RandomCrop(imagesize),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=self.transform_mean, std=self.transform_std)
                                                 ] if split == 'train'
                                                else [transforms.Resize(resize),
                                                      transforms.CenterCrop(imagesize),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=self.transform_mean,
                                                                           std=self.transform_std)
                                                      ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])

        self.imagesize = (3, imagesize, imagesize)

        self.data_to_iterate = self.get_image_data()

    def __getitem__(self, idx):
        classname, label, image_path, mask_path, imagename = self.data_to_iterate[idx]
        mode = 'L' if self.gray else 'RGB'
        image = Image.open(image_path).convert(mode)
        image = self.transform_img(image)

        if self.split == 'test' and mask_path is not None:
            mask = Image.open(mask_path).convert('L')
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            'class_id': self.class2label[classname],
            'classname': classname,
            'name': imagename,
            "mask": mask,
            "label": label
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        data_to_iterate = []
        if self.split in ('train', 'val'):
            for classname in self.class2label:
                classpath = os.path.join(self.source, classname, 'train', 'good')
                imgs = os.listdir(classpath)
                imgs = [os.path.join(classpath, x) for x in imgs if x.endswith('.png')]
                tmp = [[None for _ in range(5)] for _ in
                       range(len(imgs))]  # 五元组为（classname, label, imgpath, maskpath, imagename）
                for _ in range(len(imgs)):
                    tmp[_][0] = classname
                    tmp[_][1] = 0
                    tmp[_][2] = imgs[_]
                    tmp[_][4] = os.path.split(imgs[_])[-1]
                if self.split == 'train':
                    data_to_iterate.extend(tmp[:int(0.8*len(imgs))])
                else:
                    data_to_iterate.extend(tmp[int(0.8*len(imgs)):])

        elif self.split == 'test':
            for classname in self.class2label:
                classpath = os.path.join(self.source, classname, self.split)
                types = os.listdir(classpath)
                for typ in types:
                    imgpaths = os.listdir(os.path.join(classpath, typ))
                    imgs = [os.path.join(classpath, typ, x) for x in imgpaths if x.endswith('.png')]

                    tmp = [[None for _ in range(5)] for _ in
                           range(len(imgs))]  # 五元组为（classname, label, imgpath, maskpath, imagename）
                    for _ in range(len(imgs)):
                        img_name = os.path.split(imgs[_])[-1]
                        tmp[_][0] = classname
                        tmp[_][1] = 0 if typ == 'good' else 1
                        tmp[_][2] = imgs[_]
                        tmp[_][3] = None if typ == 'good' else os.path.join(self.source, classname,
                                                                            'ground_truth', typ,
                                                                            img_name[:3] + '_mask.png')
                        tmp[_][4] = typ + '_' + img_name

                    data_to_iterate.extend(tmp)

        else:
            raise KeyError(f"{self.split} not in ('train', 'val', 'test')")

        return data_to_iterate



if __name__ == '__main__':
    data = MVTecDataset(split='train')
    print(len(data))
