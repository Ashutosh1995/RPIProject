# RPISegProject
Repository containing code to do performance modelling on RPI

## generate_json.py
This file generates the json files by changing the hyperparamters in the defined model architecture.

## load_keras_model_run_time_rpi.py
This file loads a random json file from a list of json files, loads the keras model architecture, generates a protobuffer and then converts the protobuffer to tflite for getting the inference time. 
