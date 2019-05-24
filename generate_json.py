import json
import keras
from collections import OrderedDict
from keras import initializers
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from keras import regularizers
from keras.constraints import max_norm

channel_first_conv2d = [2**x for x in range(3,14,1)]

channel_second_conv2d = [2**x for x in range(4,16,1)]

activation = ['relu','elu','selu','tanh','exponential','linear']

kernel_initializer = ['zeros','ones','constant','random_normal','random_uniform','glorot_normal','glorot_uniform','he_normal']

bias_initializer = ['zeros','ones','constant','random_normal','random_uniform','glorot_normal','glorot_uniform','he_normal']

kernel_regularizer = ['regularizers.l1(0.)','regularizers.l2(0.)','regularizers.l1_l2(l1=0.01,l2=0.01)']

bias_regularizer =   ['regularizers.l1(0.)','regularizers.l2(0.)','regularizers.l1_l2(l1=0.01,l2=0.01)']

activity_regularizer = ['regularizers.l1(0.)','regularizers.l2(0.)','regularizers.l1_l2(l1=0.01,l2=0.01)']
var = 0
for i in channel_first_conv2d:
        for k in activation:
            for m in kernel_initializer:
                for n in bias_initializer:
                    for o in kernel_regularizer:
                        for p in bias_regularizer:
                            for q in activity_regularizer:
                                my_dict = {}
                                my_dict = {
                                        "model": "Sequential",
                                        "layers": [
                                               {
                                                "L1": "Conv2D(filters = {}, kernel_size=(3,3), strides=(1, 1), padding='valid', data_format='channels_last', activation='{}', use_bias=True, kernel_initializer='{}', bias_initializer='{}', kernel_regularizer={}, bias_regularizer={}, activity_regularizer={}, kernel_constraint={}, bias_constraint={}, input_shape=(28,28,1))".format(i,k,m,n,o,p,q,'max_norm(2.)','max_norm(2.)'),

                                                "L2": "Conv2D(filters = {}, kernel_size=(3,3), strides=(1, 1), padding='valid', data_format='channels_last', activation='{}', use_bias=True, kernel_initializer='{}', bias_initializer='{}', kernel_regularizer={}, bias_regularizer={}, activity_regularizer={}, kernel_constraint={}, bias_constraint={})".format(i,k,m,n,o,p,q,'max_norm(2.)','max_norm(2.)'),

                                                "L3": "Flatten()",

                                                "L4": "Dense(10, activation='softmax', use_bias=True, kernel_initializer='{}', bias_initializer='{}', kernel_regularizer={}, bias_regularizer={}, activity_regularizer={}, kernel_constraint={}, bias_constraint={})".format(m,n,o,p,q,'max_norm(2.)','max_norm(2.)')
                                                }
                                                   ]
                                           }
                                link = 'model' + '_' + str(var) + '.json'
                                if var <= 1000:
                                    with open(link, 'w') as fp:
                                        #data = json.loads(my_dict, object_pairs_hook=OrderedDict)
                                        json.dump(my_dict, fp, indent=4, sort_keys=False)
                                else:
                                    break
                                var += 1


print(var)
