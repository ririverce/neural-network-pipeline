import os
import json



class Cifar10Loader:

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



def cifar10_convertor(src_dir, dst_dir):
    import os
    import pickle
    import json
    import cv2

    train_labels = []
    src_file_list = ["data_batch_1",
                     "data_batch_2",
                     "data_batch_3",
                     "data_batch_4",
                     "data_batch_5"]
    for src_file in src_file_list:
        src_file_path = os.path.join(src_dir, src_file)
        with open(src_file_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')    
        label_list = data[b"labels"]
        image_list = data[b"data"]
        image_name_list = data[b"filenames"]
        for image, label, name in zip(image_list, label_list, image_name_list):
            image = image.reshape((3, 32, 32))
            image = image.transpose((1, 2, 0))
            image = image[:, :, ::-1]
            name = name.decode('utf-8')
            write_path = os.path.join(os.path.join(dst_dir, src_file),
                                      name)
            cv2.imwrite(write_path, image)
            train_labels.append({"image_path" : os.path.join("./"+src_file, name),
                                 "label" : label})
    with open(os.path.join(dst_dir, "train.json"), "w") as f:
        json.dump(train_labels, f)

    test_labels = []
    src_file_list = ["test_batch"]
    for src_file in src_file_list:
        src_file_path = os.path.join(src_dir, src_file)
        with open(src_file_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')    
        label_list = data[b"labels"]
        image_list = data[b"data"]
        image_name_list = data[b"filenames"]
        for image, label, name in zip(image_list, label_list, image_name_list):
            image = image.reshape((3, 32, 32))
            image = image.transpose((1, 2, 0))
            image = image[:, :, ::-1]
            name = name.decode('utf-8')
            write_path = os.path.join(os.path.join(dst_dir, src_file),
                                      name)
            cv2.imwrite(write_path, image)
            test_labels.append({"image_path" : os.path.join("./"+src_file, name),
                                "label" : label})
    with open(os.path.join(dst_dir, "test.json"), "w") as f:
        json.dump(test_labels, f)



def test():
    root_dir = "../dataset/bengaliai-cv19/"
    loader = BengaliAILoader(root_dir)
    dataset = loader.load()
    print(len(dataset))
    print(dataset[:5])
    

if __name__ == "__main__":
    test()
