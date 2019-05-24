import os
import time
import json
import glob
import random
import keras.layers
import numpy as np
from PIL import Image
import tensorflow as tf
import keras.backend as K
from keras import regularizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.constraints import max_norm
from keras.initializers import glorot_uniform
from keras.layers import Dense, Flatten, Conv2D

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def get_json_files():
    json_files = glob.glob('*.json')
    model_files = sorted(json_files, key=lambda name: int(name[6:-5]))
    randomly_json_file = random.choice(model_files)
    print(randomly_json_file)
    #random_pb_file = random_json_file.replace('.json','.pb')
    return randomly_json_file, model_files

def build_model(incoming_json_file):
    with open(str(incoming_json_file),'r') as fb:
        con = json.load(fb)

    model = Sequential()
    model.add(eval(con['layers'][0]['L1']))
    model.add(eval(con['layers'][0]['L2']))
    model.add(eval(con['layers'][0]['L3']))
    model.add(eval(con['layers'][0]['L4']))
    initial_weights = model.get_weights()

    backend_name = K.backend()
    if backend_name == 'tensorflow':
        k_eval = lambda placeholder: placeholder.eval(session=K.get_session())
    elif backend_name == 'theano':
        k_eval = lambda placeholder: placeholder.eval()
    else:
        raise ValueError("Unsupported backend")

    new_weights = [k_eval(glorot_uniform()(w.shape)) for w in initial_weights]

    model.set_weights(new_weights)
    input_names=[out.op.name for out in model.inputs]
    output_names=[out.op.name for out in model.outputs]
    return model, input_names, output_names

def get_pb_file(model, randomly_pb_file):
    frozen_graph = freeze_session(K.get_session(),output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, ".", randomly_pb_file, as_text=False)
    path_to_model_pb = os.getcwd() + '/' + str(randomly_pb_file)
    return path_to_model_pb


def convert_to_tflite(pb_model_path, tflite_model_name, input_name, output_name):
    mycmd = 'tflite_convert --output_file={} --graph_def_file={} --input_arrays={} --output_arrays={}'.format((tflite_model_name),pb_model_path,str(input_name[0]),str(output_name[0]))
    os.system(mycmd)
    path_to_model_tflite = os.getcwd() + '/' + str(tflite_model_name)
    return path_to_model_tflite

def get_inference_time_rpi(model_tflite, ip_name, op_name):
    (_,_),(test_image,test_label) = mnist.load_data()
    test_images = test_image[10:20]
    test_labels = test_label[10:20]
    inf_time = []
    interpreter = tf.contrib.lite.Interpreter(model_path=model_tflite)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    for i in range(test_images.shape[0]):
        img = Image.fromarray(np.asarray(test_images[i],dtype="uint8"))
        image = np.asarray(img, dtype=np.float32)
        if len(image.shape) == 2:
            input_data = image[np.newaxis, :, :, np.newaxis]
        elif len(image.shape) == 3:
            input_data = image[np.newaxis, image.shape[2], :, :]
        interpreter.set_tensor(input_details[0]['index'], input_data)
        start = time.time()
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        inf_time.append(time.time() - start)
    #print("Time: {} m.seconds/Img".format(np.mean(inf_time)*1000 , 10))
    return np.mean(inf_time)

if __name__ == '__main__':
    random_json_file, all_json_files = get_json_files()
    random_pb_file = random_json_file.replace('.json','.pb')
    random_tflite_model = random_json_file.replace('.json','.tflite')
    returned_model, inp_name, out_name = build_model(random_json_file)
    pb_model_path = get_pb_file(returned_model, random_pb_file)
    tflite_model_path = convert_to_tflite(pb_model_path, random_tflite_model, inp_name, out_name)
    inference_time = get_inference_time_rpi(tflite_model_path, inp_name, out_name)
    print(inference_time)
