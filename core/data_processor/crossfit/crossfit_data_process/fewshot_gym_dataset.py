import os
import numpy as np

class FewshotGymDataset():
    def write_to_tsv(self, lst, out_file):
        with open(out_file, "w", encoding="utf-8") as fout:
            for line in lst:
                fout.write("{}\t{}\n".format(line[0], line[1]))

class FewshotGymClassificationDataset(FewshotGymDataset):

    def get_train_test_lines(self, dataset):
        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "validation")
        return train_lines, test_lines

    def get_data(self):
        # load dataset
        dataset = self.load_dataset()

        # formulate into list (for consistency in np.random)
        train_lines, test_lines = self.get_train_test_lines(dataset)

        return train_lines, test_lines

class FewshotGymTextToTextDataset(FewshotGymDataset):

    def get_train_test_lines(self, dataset):
        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "validation")
        return train_lines, test_lines

    def get_data(self):
        """
        generate a k-shot (k) dataset using random seed (seed)
        return train, dev, test
        """

        # load dataset
        dataset = self.load_dataset()

        # formulate into list (for consistency in np.random)
        train_lines, test_lines = self.get_train_test_lines(dataset)

        return train_lines, test_lines