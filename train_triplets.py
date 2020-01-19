
import os
from math import ceil
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Dense, Activation, concatenate, Lambda, Layer
from keras import backend as K
from keras.models import load_model, Model
import keras.optimizers as optimizers
import efficientnet.keras as efn
from efficientnet.keras import EfficientNetB3
from datasets import generator_batch_triplet, get_train_df, get_val_df, center_crop

MARGIN=0.2
EMBEDDINGSIZE=1024
IMG_SIZE=96
LEARNING_RATE = 0.0001
NBR_EPOCHS=50
INITIAL_EPOCH=0
img_w, img_h=96,96
batch_size=32
evaluate_every = 1000 # interval for evaluating on one-shot tasks
num_iter = 10000 # No. of training iterations


def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)


def triplet_loss(vects):
    # f_anchor.shape = (batch_size, 256)
    f_anchor, f_positive, f_negative = vects
    # L2 normalize anchor, positive and negative, otherwise,
    # the loss will result in ''nan''!
    f_anchor = K.l2_normalize(f_anchor, axis = -1)
    f_positive = K.l2_normalize(f_positive, axis = -1)
    f_negative = K.l2_normalize(f_negative, axis = -1)

    dis_anchor_positive = K.sum(K.square(K.abs(f_anchor - f_positive)),
                                          axis = -1, keepdims = True)

    dis_anchor_negative = K.sum(K.square(K.abs(f_anchor - f_negative)),
                                         axis = -1, keepdims = True)
    loss =K.sum(K.maximum( dis_anchor_positive + MARGIN - dis_anchor_negative,0), axis=0)
    return loss

# class TripletLossLayer(Layer):
#     def __init__(self, **kwargs):
#         #self.alpha=alpha
#         super(TripletLossLayer,self).__init__(**kwargs)
#
#     def triplet_loss(self, inputs):
#         anchor, postive, negative=inputs
#         p_dist=K.sum(K.square(anchor-postive), axis=-1)
#         n_dist=K.sum(K.square(anchor-negative), axis=-1)
#         return K.sum(K.maximum(p_dist-n_dist+0.2, 0), axis=0)
#
#     def call(self,inputs):
#         loss=self.triplet_loss(inputs)
#         self.add_loss(loss)
#         return loss


def get_triplet_branch(img_w,img_h):
    attributes_branch=load_model('./working/Baseline.h5')
    attributes_branch.summary()
    attributes_branch.get_layer(name='avg_pool').name='f_base'
    f_base=attributes_branch.get_layer(name='f_base').output


    anchor=attributes_branch.input
    positive=Input(shape=(img_w,img_h,3),name='positive')
    negative=Input(shape=(img_w,img_h,3),name='negative')

    f_acs=attributes_branch.get_layer(name='f_acs').output
    f_class=attributes_branch.get_layer(name='predict_class').output

    f_sls1=Dense(1024, name='sls1')(f_base)
    f_sls2=concatenate([f_sls1,f_acs], axis=-1, name='sls1_concatenate')
    f_sls2=Dense(1024, name='sls2')(f_sls2)
    f_sls2=Activation('relu',name='sls2_relu')(f_sls2)
    f_sls3=Dense(EMBEDDINGSIZE, name='sls3')(f_sls2)
    l2_norm_output = Lambda(lambda x: K.l2_normalize(x, axis=1), name='embedding_layer')(f_sls3)
    sls_branch=Model(attributes_branch.input,l2_norm_output)

    f_anchor = sls_branch(anchor)
    f_positive = sls_branch(positive)
    f_negative = sls_branch(negative)

    loss = Lambda(triplet_loss, output_shape=(1,))([f_anchor, f_positive, f_negative])

    model = Model(inputs=[anchor, positive, negative], outputs=[f_class, loss])
    model.summary()

    return model

if __name__=='__main__':

    root_path = '/home/mary/AI/data/inaturalist-2019-fgvc6'
    model=get_triplet_branch(IMG_SIZE,IMG_SIZE)
    optimizer=SGD(lr=LEARNING_RATE, momentum=0.9, decay=0.0, nesterov=True)

    model.compile(optimizer=optimizer, metrics=["accuracy"],loss= ["categorical_crossentropy",identity_loss])
    # model.compile(loss= identity_loss,
    #               optimizer=optimizer, metrics=["accuracy"])
    model.summary()

    model_file_saved="./weights/Triplet_epoch={epoch:04d}-loss={loss:.4f}.h5"
    checkpoint=ModelCheckpoint(model_file_saved,verbose=1)
    # reduce_lr = ReduceLROnPlateau(monitor='val_'+monitor_index, factor=0.5,
    #               patience=5, verbose=1, min_lr=0.00001)
    # early_stop = EarlyStopping(monitor='val_'+monitor_index, patience=15, verbose=1)

    print ("start training.................")

    # for i in range (1,num_iter+1):
    #     triplets_batch=get_triplets_batch(root_path,batch_size, img_w, img_h)
    #     [class_loss, triplet_loss]=model.train_on_batch(triplets_batch, None)
    #
    #     if i % evaluate_every == 0:
    #         print("\n ------------- \n")
    #         print("{0} iterations: , classification Loss: {1}, triplet loss: {2}".format(i,class_loss, triplet_loss))
    #         #probs, yprob = compute_probs(base, x_test_ori[:n_val, :, :, :], y_test_ori[:n_val])

    train_df, nb_categories=get_train_df(root_path)
    val_df=get_val_df(root_path)
    steps_per_epoch = int(ceil(train_df.shape[0] * 1. / batch_size))
    validation_steps= int(ceil(val_df.shape[0] * 1. / batch_size))


    model.fit_generator(generator_batch_triplet(root_path, batch_size, train_df, val_df, nb_categories, img_w, img_h,
                            crop_method=center_crop, scale_ratio=1.0, random_scale=True, mode='train',
                            return_label=True, shuffle=True),
                        steps_per_epoch=steps_per_epoch, epochs=NBR_EPOCHS, verbose=1,
                        validation_data=generator_batch_triplet(root_path, batch_size, train_df, val_df, nb_categories, img_w, img_h,
                            crop_method=center_crop, scale_ratio=1.0, random_scale=True, mode='val',
                            return_label=True, shuffle=False),
                        validation_steps=validation_steps,
                        callbacks=[checkpoint], initial_epoch =INITIAL_EPOCH,
                        max_queue_size=100, workers=1, use_multiprocessing=False)
