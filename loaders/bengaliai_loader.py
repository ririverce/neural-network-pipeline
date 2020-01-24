import os
import json



class BengaliAILoader:

    train_label_file = "train.json"
    test_label_file = "test.json"

    def __init__(self, root_dir):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, self.train_label_file), "r") as f:
            self.train_labels = json.load(f)
            for i in range(len(self.train_labels)):
                self.train_labels[i]["image_path"] = os.path.join(
                                                         self.root_dir,
                                                         self.train_labels[i]["image_path"])
        with open(os.path.join(self.root_dir, self.test_label_file), "r") as f:
            self.test_labels = json.load(f)
            for i in range(len(self.test_labels)):
                self.test_labels[i]["image_path"] = os.path.join(
                                                        self.root_dir,
                                                        self.test_labels[i]["image_path"])

    def load(self):
        dataset = self.train_labels + self.test_labels
        return dataset



def bengaliai_convertor(src_dir, dst_dir):
    import os
    import json
    import numpy as np
    import cv2
    import pandas as pd

    train_df = pd.read_csv(os.path.join(src_dir, 'train.csv'))
    train_labels = []
    train_image_files = ["train_image_data_0.parquet",
                         "train_image_data_1.parquet",
                         "train_image_data_2.parquet",
                         "train_image_data_3.parquet"]
    save_train_image_dirs = [f.split(".")[0] for f in train_image_files]
    train_image_files = [os.path.join(src_dir, f) for f in train_image_files]
    save_train_image_dirs = [os.path.join(dst_dir, f) for f in save_train_image_dirs]
    for sf, dd in zip(train_image_files, save_train_image_dirs):
        image_df = pd.read_parquet(sf)
        for row in image_df.itertuples():
            file_name = os.path.join(dd, row[1]+".png")
            file_name = os.path.join(dd.split("/")[-1], row[1]+".png")
            image_data = np.array(row[2:])
            image_data = image_data.reshape((137, 236))
            image_data = image_data.astype(np.uint8)
            cv2.imwrite(file_name, image_data)
            label_df = train_df[train_df["image_id"]==row[1]]
            if len(label_df.iloc[0]) == 5:
                _, grapheme_root, vowel_diacritic, consonant_diacritic, grapheme = label_df.iloc[0]
            else:
                label_df.iloc[0]
            train_labels.append({"image_path" : file_name,
                                 "grapheme_root" : int(grapheme_root),
                                 "vowel_diacritic" : int(vowel_diacritic),
                                 "consonant_diacritic" : int(consonant_diacritic),
                                 "grapheme" : grapheme})
    with open(os.path.join(dst_dir, "train.json"), "w") as f:
        json.dump(train_labels, f)

    test_df = pd.read_csv(os.path.join(src_dir, 'test.csv'))
    test_labels = []
    test_image_files = ["test_image_data_0.parquet",
                        "test_image_data_1.parquet",
                        "test_image_data_2.parquet",
                        "test_image_data_3.parquet"]
    save_test_image_dirs = [f.split(".")[0] for f in test_image_files]
    test_image_files = [os.path.join(src_dir, f) for f in test_image_files]
    save_test_image_dirs = [os.path.join(dst_dir, f) for f in save_test_image_dirs]
    for sf, dd in zip(test_image_files, save_test_image_dirs):
        image_df = pd.read_parquet(sf)
        for row in image_df.itertuples():
            file_path = os.path.join(dd, row[1]+".png")
            file_name = os.path.join(dd.split("/")[-1], row[1]+".png")
            image_data = np.array(row[2:])
            image_data = image_data.reshape((137, 236))
            image_data = image_data.astype(np.uint8)
            cv2.imwrite(file_path, image_data)
            label_df = test_df[test_df["image_id"]==row[1]]
            if len(label_df.iloc[0]) == 5:
                _, grapheme_root, vowel_diacritic, consonant_diacritic, grapheme = label_df.iloc[0]
            else:
                label_df.iloc[0]
            test_labels.append({"image_path" : file_name,
                                 "grapheme_root" : int(grapheme_root),
                                 "vowel_diacritic" : int(vowel_diacritic),
                                 "consonant_diacritic" : int(consonant_diacritic),
                                 "grapheme" : grapheme})
    with open(os.path.join(dst_dir, "test.json"), "w") as f:
        json.dump(test_labels, f)




def test():
    root_dir = "../dataset/bengaliai-cv19/"
    loader = BengaliAILoader(root_dir)
    dataset = loader.load()
    print(len(dataset))
    print(dataset[:5])
    #bengaliai_convertor("/media/hal/95e63264-ad11-4f6d-acc3-449ff70a23db/dataset/kaggle-raw/bengaliai-cv19/",
    #                    "/media/hal/95e63264-ad11-4f6d-acc3-449ff70a23db/dataset/kaggle-preprocessed/bengaliai-cv19/")


if __name__ == "__main__":
    test()