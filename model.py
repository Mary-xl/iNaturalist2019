

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.inception_v3 import InceptionV3

def get_model(img_w,img_h,num_classes):

    inceptionv3 = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(img_w, img_h, 3),
                              pooling='avg')
    f_base = inceptionv3.get_layer(index=-1).output
    output = Dense(1024, activation='relu', name='')(f_base)
    output = Dropout(0.5)(output)
    predictions = Dense(num_classes, activation='softmax', name='predict_class')(output)
    model = Model(inputs=inceptionv3.input, outputs=predictions)
    return model