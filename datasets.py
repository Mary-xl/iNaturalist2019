import os
import json
import pandas as pd

def get_train_df(root_path):

    train_file=os.path.join(root_path,'train2019.json')
    with open(train_file) as data_file:
        train_file = json.load(data_file)
    train_anns_df = pd.DataFrame(train_file['annotations']).drop(columns="id")
    train_img_df = pd.DataFrame(train_file['images'])[['id', 'file_name']].rename(columns={'id': 'image_id'})
    train_data_df=pd.merge(train_anns_df, train_img_df, on='image_id')
    train_data_df['category_id']=train_data_df['category_id'].astype(str)
    num_classes=len(train_data_df['category_id'].unique())
    return train_data_df, num_classes

def get_val_df(root_path):

    valid_file = os.path.join(root_path,'val2019.json')
    with open(valid_file) as data_file:
        valid_file = json.load(data_file)
    valid_anns_df = pd.DataFrame(valid_file['annotations'])[['image_id', 'category_id']]
    valid_img_df = pd.DataFrame(valid_file['images'])[['id', 'file_name']].rename(columns={'id': 'image_id'})
    valid_data_df = pd.merge(valid_anns_df, valid_img_df, on="image_id")
    valid_data_df['category_id'] = valid_data_df['category_id'].astype(str)
    return valid_data_df
