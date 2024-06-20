import os
from torch.utils.data import Dataset
import torch 
import cv2

"""
Custom Dataset class
"""
class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.data = data
        self.length = len(self.data)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

def get_data():

    train_folder = "./FER/train"
    test_folder = "./FER/test"
    emotions = os.listdir(train_folder)
    emotion_map = {}
    for i, emotion in enumerate(emotions):
        emotion_map[emotion] = i
    print(emotion_map)

    """
    load data is a lambda function
    It goes to all the folders/emotions in the input folder (x),
    then it goes to each image in each emotion folder and does the following
        → load the image 
        → convert from numpy array to torch tensor 
        → change datatype to float 
        → unsqueeze(Add extra dimension) 
        → divide by 255.0 to normalise the input
        -------------------------------------------
        → convert the emotion to the number using emotion map

    Package imput and output to a list and put this in a list.  
    """

    load_data = lambda x :  [
                                [
                                    torch.from_numpy(
                                        cv2.imread(
                                            os.path.join(x,emotion, name),
                                            cv2.IMREAD_GRAYSCALE
                                        )
                                    ).float().unsqueeze(0)/255.0, 
                                    emotion_map[emotion]
                                ]
                                for emotion in os.listdir(x)
                                for name in os.listdir(os.path.join(x, emotion))
                            ]
    
    train_data = MyDataset(load_data(train_folder))
    test_data = MyDataset(load_data(test_folder))
    print(len(train_data), len(test_data))
    return train_data, test_data, emotion_map

