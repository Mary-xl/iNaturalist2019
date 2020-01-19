import os
import json
import pandas as pd
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import preprocess_input
from sklearn.utils import shuffle as df_shuffle

MARGIN=0.2
EMBEDDINGSIZE=1024
IMG_SIZE=96
BATCH_SIZE = 32
classes=['Amphibians','Birds','Fungi','Insects','Plants','Reptiles']

def get_train_df(root_path):

    train_file=os.path.join(root_path,'train2019.json')
    with open(train_file) as data_file:
        train_file = json.load(data_file)
    train_anns_df = pd.DataFrame(train_file['annotations']).drop(columns="id")
    train_img_df = pd.DataFrame(train_file['images'])[['id', 'file_name']].rename(columns={'id': 'image_id'})
    train_data_df=pd.merge(train_anns_df, train_img_df, on='image_id')
    train_data_df['category_id']=train_data_df['category_id'].astype(str)

    train_data_df['class']=train_data_df['file_name'].str.split('/', expand = True)[1]

    num_categories=len(train_data_df['category_id'].unique())
    return train_data_df, num_categories

def get_val_df(root_path):

    valid_file = os.path.join(root_path,'val2019.json')
    with open(valid_file) as data_file:
        valid_file = json.load(data_file)
    valid_anns_df = pd.DataFrame(valid_file['annotations'])[['image_id', 'category_id']]
    valid_img_df = pd.DataFrame(valid_file['images'])[['id', 'file_name']].rename(columns={'id': 'image_id'})
    valid_data_df = pd.merge(valid_anns_df, valid_img_df, on="image_id")
    valid_data_df['category_id'] = valid_data_df['category_id'].astype(str)
    valid_data_df['class'] = valid_data_df['file_name'].str.split('/', expand=True)[1]

    return valid_data_df

#def generate_triplets(generator, batch_size,img_h,img_w):


def center_crop(x, center_crop_size):
    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    cropped = x[centerw - halfw : centerw + halfw,
                 centerh - halfh : centerh + halfh, :]

    return cropped

def scale_byRatio(img_path, ratio=1.0, return_width=IMG_SIZE, crop_method=center_crop):
    # Given an image path, return a scaled array
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    shorter = min(w, h)
    longer = max(w, h)
    img_cropped = crop_method(img, (shorter, shorter))
    img_resized = cv2.resize(img_cropped, (return_width, return_width),
                            interpolation=cv2.INTER_CUBIC)
    img_rgb = img_resized
    img_rgb[:, :, [0, 1, 2]] = img_resized[:, :, [2, 1, 0]]

    return img_rgb


def generator_batch_triplet(root_path,batch_size,train_df,val_df, nbr_categories, img_w, img_h,crop_method=center_crop, scale_ratio=1.0, random_scale=False, mode='train',return_label=True, shuffle=False):

    if mode=='train':
       X_df=train_df
    else:
       X_df=val_df
    # if shuffle:
    #     random.shuffle(X_df)
    #X_df=X_df[:100]
    N=X_df.shape[0]
    batch_index=0
    while True:

        current_index=(batch_index*batch_size)%N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        # 如果一轮epoch结束
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
               X_df=df_shuffle(X_df)
            continue

        print("  batch id:", batch_index)

        X_anchor = np.zeros((batch_size, img_w, img_h, 3))
        X_positive = np.zeros((batch_size, img_w, img_h, 3))
        X_negative = np.zeros((batch_size, img_w, img_h, 3))

        Y_batch_one = np.zeros((batch_size, nbr_categories))
        Y_batch_fake = np.zeros((batch_size, 1))

        for i in range(current_index, current_index + current_batch_size):

            if random_scale:
              scale_ratio = random.uniform(0.9, 1.1)
            row=X_df.iloc[i]
            anchor_category=row['category_id']
            anchor_class=row['class']
            anchor_path=os.path.join(root_path,str(row['file_name']))
            anchor = scale_byRatio(anchor_path, ratio=scale_ratio, return_width=img_w,crop_method=crop_method)

            if mode=='train':
                 anchor_df = X_df.loc[X_df['category_id'] == anchor_category]
                 [idx_P]=np.random.choice(anchor_df.shape[0], size=1, replace=False)
                 positive_path=os.path.join(root_path,str(anchor_df.iloc[idx_P]['file_name']))
                 positive = scale_byRatio(positive_path, ratio=scale_ratio, return_width=img_w, crop_method=crop_method)

                 class_df=X_df.loc[X_df['class'] == anchor_class]
                 negative_df=class_df[class_df.category_id!=anchor_category]
                 [idx_N]=np.random.choice(negative_df.shape[0],size=1,replace=False)
                 negative_path=os.path.join(root_path,str(negative_df.iloc[idx_N]['file_name']))
                 negative = scale_byRatio(negative_path, ratio=scale_ratio, return_width=img_w, crop_method=crop_method)

                 X_anchor[i-current_index] = anchor
                 X_positive[i-current_index] = positive
                 X_negative[i-current_index] = negative

            elif mode=='val':
                 X_anchor[i - current_index] = anchor
                 X_positive[i - current_index] = anchor
                 X_negative[i - current_index] = anchor

            if return_label:
                 label_one = int(anchor_category)
                 Y_batch_one[i - current_index, label_one] = 1

        # triplets_batch=[X_anchor,X_positive,X_negative]
        # display_triplets(triplets_batch,batch_size)

        
        X_anchor = X_anchor.astype(np.float64) # 转为浮点数
        X_positive = X_positive.astype(np.float64)
        X_negative = X_negative.astype(np.float64)

        X_anchor = preprocess_input(X_anchor) # 标准化
        X_positive = preprocess_input(X_positive)
        X_negative = preprocess_input(X_negative)


        if return_label:
            yield ([X_anchor, X_positive, X_negative],[Y_batch_one,Y_batch_fake])
        else:
            yield [X_anchor, X_positive, X_negative]

        # pick_class = classes[i]
        # class_df=X_df.loc[X_df['class'] == pick_class]
        # categories=class_df['category_id'].unique()
        # num_categories=len( categories)
        # anchor_category=categories[np.random.randint(0, num_categories)]
        # anchor_df=class_df.loc[class_df['category_id'] ==anchor_category]
        # [idx_A, idx_P]=np.random.choice(anchor_df.shape[0],size=2,replace=False)
        #
        # anchor_path=os.path.join(root_path,str(anchor_df.iloc[idx_A]['file_name']))
        # anchor = scale_byRatio(anchor_path, ratio=scale_ratio, return_width=img_w,
        #                          crop_method=crop_method)



           #positive_path=os.path.join(root_path,str(anchor_df.iloc[idx_P]['file_name']))
           #positive = scale_byRatio(positive_path, ratio=scale_ratio, return_width=img_w,
           #                      crop_method=crop_method)

           #negative_df=class_df[class_df.category_id!=anchor_category]
           #[idx_N]=np.random.choice(negative_df.shape[0],size=1,replace=False)
           #negative_path=os.path.join(root_path,str(negative_df.iloc[idx_N]['file_name']))
           #negative = scale_byRatio(negative_path, ratio=scale_ratio, return_width=img_w,
           #                      crop_method=crop_method)

def display_triplets(triplets_batch,batch_size):

    b=triplets_batch[0].shape[0]
    labels=['Anchor','Positive', 'Negative']

    for i in range(batch_size):
        fig=plt.figure(figsize=(16,2))
        for j in range(3):
            subplot=fig.add_subplot(1,3,j+1)
            img=triplets_batch[j][i,:,:,:]
            img = np.ndarray.astype(img, np.uint8)
            plt.imshow(img)
            subplot.title.set_text(labels[j])
        plt.show()



#hard_size: number of hard triplets in a batch; random_size: number of random triplets in a batch
#hard_size+random_size=batch_size
# def get_hard_triplet_batch(dataset_train, dataset_test,size, hard_size,random_size,network, s):
#
#     if s=='train':
#         X=dataset_train
#     else:
#         X=dataset_test
#
#     _,w,h,c=X[0].shape
#
#     random_batch=get_random_triplet(dataset_train, dataset_test,size,s)
#     random_batch_loss=np.zeros((size)) #initialize loss for the random batch
#
#     A=network.predict(random_batch[0])
#     P=network.predict(random_batch[1])
#     N=network.predict(random_batch[2])
#
#     random_batch_loss=np.sum(np.square(A-P),axis=1)-np.sum(np.square(A-N),axis=1)
#     #sort the loss by distance, the higher the harder, and select the hardest hard_size samples
#     hard_select=np.argsort(random_batch_loss)[::-1][:hard_size]
#     random_select=np.random.choice(np.delete(np.arange(size),hard_select),random_size, replace=False)
#     selection=np.append(hard_select,random_select)
#
#     triplets=[random_batch[0][selection,:,:,:], random_batch[1][selection,:,:,:], random_batch[2][selection,:,:,:]]
#
#     return triplets


if __name__=='__main__':

    root_path='/home/mary/AI/data/inaturalist-2019-fgvc6'
    nbr_categories=1010
    train_df, num_categories=get_train_df(root_path)

    val_df=get_val_df(root_path)
   # generator_batch_triplet(root_path, BATCH_SIZE,train_df,val_df, nbr_categories, IMG_SIZE, IMG_SIZE)
    generator_batch_triplet(root_path,BATCH_SIZE,train_df,val_df, nbr_categories, IMG_SIZE, IMG_SIZE)
