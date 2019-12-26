

from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.applications.inception_v3 import InceptionV3
import efficientnet.keras as efn
from efficientnet.keras import EfficientNetB3

def get_model(img_w,img_h,num_classes):

    # inceptionv3 = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(img_w, img_h, 3),
    #                           pooling='avg')
    # f_base = inceptionv3.get_layer(index=-1).output

    f_base=EfficientNetB3( weights = 'imagenet', include_top = False, input_shape = (img_w, img_h, 3))
    #output=f_base.get_layer(index=-1).output
    output=Flatten()(f_base.output)
    output = Dense(1024, activation='relu', name='')(output)
    output = Dropout(0.5)(output)
    predictions = Dense(num_classes, activation='softmax', name='predict_class')(output)
    model = Model(inputs=f_base.input, outputs=predictions)
    return model