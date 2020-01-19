
import os
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.applications.inception_v3 import InceptionV3
import efficientnet.keras as efn
from efficientnet.keras import EfficientNetB3
from datasets import get_train_df,get_val_df
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import keras.optimizers as optimizers
import json

GPUS = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS


BATCH_SIZE=256
IMG_SIZE=96
LEARNING_RATE=0.0001
NUM_EPOCHS=1000

def get_attribute_branch(img_w,img_h,num_classes):

    inceptionv3 = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(img_w, img_h, 3),pooling='avg')
    f_base = inceptionv3.get_layer(index=-1).output
    # efficientnetB3=EfficientNetB3( weights = 'imagenet', include_top = False, input_shape = (img_w, img_h, 3), pooling='avg')
    # f_base=efficientnetB3.get_layer(index=-1).output

    #output=Flatten()(f_base)
    output = Dense(1024, activation='relu', name='f_acs')(f_base)
    output = Dropout(0.5)(output)
    predictions = Dense(num_classes, activation='softmax', name='predict_class')(output)

    model = Model(inputs=inceptionv3.input, outputs=predictions)
    model.summary()
    return model

def train_attribute_branch(root_path):
    #train_dir=os.path.join(root_path,'train_val2019')

    train_data_df, num_classes=get_train_df(root_path)
    print (train_data_df.head(), num_classes)
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       validation_split=0.25,
                                       horizontal_flip=True,
                                       zoom_range=0.3,
                                       width_shift_range=0.3,
                                       height_shift_range=0.3
                                       )
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_data_df,
        directory=root_path,
        x_col="file_name",
        y_col="category_id",
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode="categorical",
        target_size=(IMG_SIZE, IMG_SIZE))

    val_data_df=get_val_df(root_path)
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_data_df,
        directory=root_path,
        x_col="file_name",
        y_col="category_id",
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode="categorical",
        target_size=(IMG_SIZE,IMG_SIZE))

    model=get_attribute_branch(IMG_SIZE,IMG_SIZE, num_classes)
    print(model.summary())

    optimizer = SGD(lr=LEARNING_RATE, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    weightfile_path="./working/Baseline.h5"
    checkpoint = ModelCheckpoint(weightfile_path, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

    if os.path.exists(weightfile_path):
        model=load_model(weightfile_path)
    history = model.fit_generator(generator=train_generator,
                                    steps_per_epoch=5,
                                    validation_data=val_generator,
                                    validation_steps=2,
                                    epochs=NUM_EPOCHS,
                                    callbacks=[checkpoint, early],
                                    verbose=2)

    with open('./working/history.json', 'w') as f:
        json.dump(history.history, f)

if __name__=='__main__':

    root_path='/home/mary/AI/data/inaturalist-2019-fgvc6'
    train_attribute_branch(root_path)
