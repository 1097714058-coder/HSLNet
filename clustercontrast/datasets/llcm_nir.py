from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
from ..utils.data import BaseImageDataset


class llcm_nir(BaseImageDataset):
    """
    LLCM NIR Dataset
    Dataset statistics:
    # identities: 1066
    # images: training set + test set
    """
    dataset_dir = 'LLCM'

    def __init__(self, root, verbose=True, **kwargs):
        super(llcm_nir, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        
        self.train_dir = osp.join(self.dataset_dir, 'nir')
        self.query_dir = osp.join(self.dataset_dir, 'test_nir')
        self.gallery_dir = osp.join(self.dataset_dir, 'test_vis')

        self._check_before_run()

        train = self._process_dir_train(self.train_dir, relabel=True)
        # For test sets, use idx files to get the list of images
        query = self._process_dir_test_from_idx(self.query_dir, osp.join(self.dataset_dir, 'idx/test_nir.txt'))
        gallery = self._process_dir_test_from_idx(self.gallery_dir, osp.join(self.dataset_dir, 'idx/test_vis.txt'))

        if verbose:
            print("=> LLCM NIR loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir_train(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*', '*.jpg'))
        pattern = re.compile(r'(\d+)_c(\d+)_s(\d+)_f(\d+)_nir\.jpg')

        pid_container = set()
        for img_path in img_paths:
            match = pattern.search(img_path)
            if match:
                pid = int(match.group(1))
                pid_container.add(pid)
        
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
        
        dataset = []
        for img_path in img_paths:
            match = pattern.search(img_path)
            if match:
                pid = int(match.group(1))
                camid = int(match.group(2))
                if relabel:
                    pid = pid2label[pid]
                camid = camid - 1  # index starts from 0
                dataset.append((img_path, pid, camid))

        return dataset

    def _process_dir_test_from_idx(self, dir_path, idx_file):
        dataset = []
        with open(idx_file, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 1:
                        img_path = osp.join(self.dataset_dir, parts[0])
                        # Extract pid from filename
                        filename = osp.basename(img_path)
                        pattern = re.compile(r'(\d+)_c(\d+)_s(\d+)_f(\d+)_(nir|vis)\.jpg')
                        match = pattern.search(filename)
                        if match:
                            pid = int(match.group(1))
                            camid = int(match.group(2)) - 1
                            dataset.append((img_path, pid, camid))
        return dataset

    def _process_dir_test(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*', '*.jpg'))
        pattern = re.compile(r'(\d+)_c(\d+)_s(\d+)_f(\d+)_(nir|vis)\.jpg')

        pid_container = set()
        for img_path in img_paths:
            # Extract filename from path
            filename = osp.basename(img_path)
            match = pattern.search(filename)
            if match:
                pid = int(match.group(1))
                pid_container.add(pid)
        
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
        
        dataset = []
        for img_path in img_paths:
            # Extract filename from path
            filename = osp.basename(img_path)
            match = pattern.search(filename)
            if match:
                pid = int(match.group(1))
                camid = int(match.group(2))
                if relabel:
                    pid = pid2label[pid]
                camid = camid - 1  # index starts from 0
                dataset.append((img_path, pid, camid))

        return dataset
