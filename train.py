
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import backend as K
from keras import optimizers
from datasets import get_train_df,get_val_df
from model import get_model


BATCH_SIZE=256
IMG_SIZE=96
LEARNING_RATE=0.0001
NUM_EPOCHS=100


def train_baseline(root_path):
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

    model=get_model(IMG_SIZE,IMG_SIZE, num_classes)
    print(model.summary())

    optimizer = SGD(lr=LEARNING_RATE, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint("./working/Baseline.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto')

    history = model.fit_generator(generator=train_generator,
                                    steps_per_epoch=5,
                                    validation_data=val_generator,
                                    validation_steps=2,
                                    epochs=NUM_EPOCHS,
                                    callbacks=[checkpoint, early],
                                    verbose=2)

if __name__=='__main__':

    #root_path='/home/mary/AI/data/iNat2019_FGVC'
    root_path = '/data/iNat2019_FGVC'
    train_baseline(root_path)
