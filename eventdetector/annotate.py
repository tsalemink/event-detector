from eventdetector.utils import *
from urllib.request import urlretrieve

import numpy as np

import sys
# TODO: Can't get this working. Disable it for now.
# from ezc3d import c3d # Use ezc3d instead of btk
import os
import csv
import keras
from keras.models import load_model

keras.losses.weighted_binary_crossentropy = weighted_binary_crossentropy


def derivative(traj, nframes):
    traj_der = traj[1:nframes, :] - traj[0:(nframes-1), :]
    return np.append(traj_der, [[0, 0, 0]], axis=0)


def extract_kinematics(filename_in):
    print("Trying %s" % filename_in)
    
    # Open c3d and read data
    c = c3d(filename_in)
    data = c["data"]["points"]
    shape = data.shape
    nframes = shape[2]
    first_frame = 0
    labels = c['parameters']['POINT']['LABELS']['value']

    rate = c['parameters']['POINT']['RATE']['value']

    # We extract only kinematics
    kinematics = ["HipAngles", "KneeAngles", "AnkleAngles", "PelvisAngles", "FootProgressAngles"]
    markers = ["ANK", "TOE", "KNE", "ASI", "HEE"]
    
    # Cols
    # 2 * 5 * 3 = 30  kinematics
    # 2 * 5 * 3 = 30  marker trajectories
    # 2 * 5 * 3 = 30  marker trajectory derivatives
    # 3 * 3 = 9       extra trajectories

    outputs = np.array([[0] * nframes, [0] * nframes]).T
    
    # Check if there are any kinematics in the file
    preFix = 'A22' # Change accordingly to your data
    brk = True
    for i in range(5):
        if labels.index(preFix + ":L" + kinematics[i]):
            brk = False
            break

    if brk:
        print("No kinematics in %s!" % (filename,))
        return

    # Combine kinematics into one big array
    angles = [None] * (len(kinematics) * 2)
    for i, v in enumerate(kinematics):
        kinematicsIdx = labels.index(preFix + ':L' + v)
        point = data[0:3, kinematicsIdx, :]
        angles[i] = np.transpose(point)
        kinematicsIdx = labels.index(preFix + ':R' + v)
        point = data[0:3, kinematicsIdx, :]
        angles[len(kinematics) + i] = np.transpose(point)
    
    # Get the pelvis
    idxLASI = labels.index(preFix + ':LASI')
    LASI = np.transpose(data[0:3, idxLASI, :])
    idxRASI = labels.index(preFix + ':RASI')
    RASI = np.transpose(data[0:3, idxRASI, :])
    midASI = (LASI + RASI) / 2
    # incrementX = 1 if midASI[100][0] > midASI[0][0] else -1

    traj = [None] * (len(markers) * 4 + 3)
    for i, v in enumerate(markers):
        try:
            markersIdx = labels.index(preFix + ':L' + v)
            traj[i] = np.transpose(data[0:3, markersIdx, :]) - midASI
            markersIdx = labels.index(preFix + ':R' + v)
            traj[len(markers) + i] = np.transpose(data[0:3, markersIdx, :]) - midASI
        except:
            print("Error while reading marker data: %d, %s" % (i, v))
            return

        traj[i][:, 0] = traj[i][:, 0]   # * incrementX
        traj[len(markers) + i][:, 0] = traj[len(markers) + i][:, 0]  # * incrementX
        traj[i][:, 2] = traj[i][:, 2]   # * incrementX
        traj[len(markers) + i][:, 2] = traj[len(markers) + i][:, 2]  # * incrementX

    for i in range(len(markers)*2):
        traj[len(markers)*2 + i] = derivative(traj[i], nframes) 
        
    midASI = midASI     # * incrementX

    midASIvel = derivative(midASI, nframes)
    midASIacc = derivative(midASIvel, nframes)

    traj[len(markers)*4] = midASI
    traj[len(markers)*4 + 1] = midASIvel
    traj[len(markers)*4 + 2] = midASIacc

    curves = np.concatenate(angles + traj, axis=1)

    return curves


def convert_data(data):
    # TODO: temporary mess
    def derivative(traj):
        nframes = traj.shape[0]
        traj_der = traj[1:nframes, :] - traj[0:(nframes-1), :]
        return np.append(traj_der, [[0] * traj.shape[1]], axis=0)

    # We assume the data has following sequences
    # joint angles (3 DOF each):
    # - hip
    # - knee
    # - ankle
    # - pelvis
    # - foot progression
    # markers positions (3 DOF each):
    # - ankle
    # - toes
    # - knee
    # - pelvis
    # - heel
    # if there are two extra columns we assume that these are binary
    # sequences of heel strike and foot off events respectively
    if data.shape[1] != 30 and data.shape[1] != 32:
        sys.exit("Wrong data format. There should be 30 columns.")

    # What we actually use are:
    # - joint angles (5 x 3)
    # - velocity of markers (5 x 3)
    # - velocity and acceleration of the pelvis (2 x 3)
    X = np.zeros((data.shape[0], 15 + 15 + 6))
    X[:, 0:15] = data[:, 0:15]
    X[:, 15:30] = derivative(data[:, 15:30])
    X[:, 30:33] = derivative(data[:, 24:27])
    X[:, 33:36] = derivative(derivative(data[:, 24:27]))

    Y = None
    if data.shape[1] != 32:
        Y = data[:, 30:32]

    return X, Y


def neural_method(inputs, model):
    cols = list(range(15)) + [15 + i for i in range(13)] + [30 + i for i in range(6)]
    # TODO: This input is exactly the same both environments.
    #   But the outputs are quite different...
    res = model.predict(inputs[:, cols].reshape((1, inputs.shape[0], len(cols))))

    # TODO: REMOVE:
    #   The complete inputs should be the same then as well...?
    # print(f"COMPLETE-INPUT: {inputs[:, cols].reshape((1, inputs.shape[0], len(cols)))}")

    # TODO: REMOVE:
    #  res[0]: [[1.29914582e-02]
    #  [1.02045422e-03]
    #  [7.69939448e-04] (3.6)
    #   ...
    #  res[0]: [[1.51821403e-02]
    #  [1.26197957e-03]
    #  [9.16094636e-04]
    # print(f"res[0]: {res[0]}")
    # print(f"SIZE: {len(res[0])}")

    # TODO: Yeah `res[0]` is still different.
    #   `inputs.shape[0]` and `cols` consistent.
    #   `inputs` appears consistent as well.
    #   So it's just the model is processing it differently
    #   As long as they're in the same format it shouldn't matter...
    #   Why is it actually failing...???

    # TODO: This is failing in new keras versions.
    peakind = peakdet(res[0], 0.7)
    frames = list(map(int, [k for k, v in peakind[0]]))
    return frames


def get_models():
    if not os.path.exists("models/FO.h5"):
        print("Model not found. Downloading...")
        try:
            os.makedirs("models")
        except:
            pass
        model_path = "https://s3-eu-west-1.amazonaws.com/kidzinski/event-detector/FO.h5"
        urlretrieve(model_path, "models/FO.h5")
        model_path = "https://s3-eu-west-1.amazonaws.com/kidzinski/event-detector/HS.h5"
        urlretrieve(model_path, "models/HS.h5")
        print("Model downloaded!")


# # TODO: Must be disabled for old tensorflow:
# # TODO: Define model architecture from scratch.
# #   We should probably get this working in 3.6 first.
# modelFO_new = keras.models.Sequential()
# modelFO_new.add(keras.layers.Input(shape=(None, 34)))
# modelFO_new.add(keras.layers.LSTM(units=64, return_sequences=True))
# modelFO_new.add(keras.layers.LSTM(units=64, return_sequences=True))
# # TODO: Is this Dense layer correct?
# #   `1` OK, or 9/18...?
# modelFO_new.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid')))
# modelFO_new.load_weights("models/FO.h5")
# # modelFO_new.load_weights("models/FO_new.h5")
#
# modelHS_new = keras.models.Sequential()
# modelHS_new.add(keras.layers.Input(shape=(None, 34)))
# modelHS_new.add(keras.layers.LSTM(units=64, return_sequences=True))
# modelHS_new.add(keras.layers.LSTM(units=64, return_sequences=True))
# modelHS_new.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid')))
# modelHS_new.load_weights("models/HS.h5")
# # modelHS_new.load_weights("models/HS_new.h5")

# TODO: Old keras version.
#   This works in 3.6!
#   But the approach above does not work with new keras.
#   MAYBE best to just write a new model file using the fresh configuration...!
modelFO_new = keras.models.Sequential()
modelFO_new.add(keras.layers.LSTM(units=64, return_sequences=True, batch_input_shape=(None, None, 34)))
modelFO_new.add(keras.layers.LSTM(units=64, return_sequences=True))
modelFO_new.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid')))
modelFO_new.load_weights("models/FO.h5")

modelHS_new = keras.models.Sequential()
modelHS_new.add(keras.layers.LSTM(units=64, return_sequences=True, batch_input_shape=(None, None, 34)))
modelHS_new.add(keras.layers.LSTM(units=64, return_sequences=True))
modelHS_new.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid')))
modelHS_new.load_weights("models/HS.h5")

# # TODO: Check if this fresh model can actually be used.
# #   If so, try save the model(s) using this new (hopefully correct) syntax.
# #   Unfortunately this still had the 'Dense' error in keras 3.4.1...
# new_models_dir = "C:\\Users\\tsal421\\Projects\\Gait\\event-detector\\new_models"
# new_FO_path = os.path.join(new_models_dir, "FO_new.h5")
# new_HS_path = os.path.join(new_models_dir, "HS_new.h5")
# modelFO_new.save(new_FO_path)
# modelHS_new.save(new_HS_path)


# get_models()
# modelFO = load_model("models/FO.h5")
# modelHS = load_model("models/HS.h5")
# modelFO = load_model("models/FO_new.h5")
# modelHS = load_model("models/HS_new.h5")


# TODO: REMOVE:
# import json
# architecture_dict = json.loads(modelFO.to_json())
# pretty_architecture = json.dumps(architecture_dict, indent=4)
# print(f"ARCHITECTURE: {pretty_architecture}")
# # print(f"WEIGHTS: {modelFO.get_weights()}")


def process(filename_in, filename_out):
    idxL = [(int(i / 3)) * 3 + i for i in range(30)]
    idxR = [3 + (int(i / 3)) * 3 + i for i in range(30)]

    # TODO: Temporarily disable.
    # inputs = extract_kinematics(filename_in)

    # TODO: Try using manually serialised inputs:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "extracted_kinematics.json.npy")
    inputs = np.load(file_path)

    # TODO: XR is exactly the same for both environments...
    inputsL = inputs[:, idxL]
    inputsR = inputs[:, idxR]
    XL, YL = convert_data(inputsL)
    XR, YR = convert_data(inputsR)

    events = {
        # ("Foot Strike", "Left"): neural_method(XR, modelFO),
        # ("Foot Strike", "Right"): neural_method(XL, modelFO),
        # ("Foot Off", "Left"): neural_method(XR, modelHS),
        # ("Foot Off", "Right"): neural_method(XL, modelHS),
        ("Foot Strike", "Left"): neural_method(XR, modelFO_new),
        ("Foot Strike", "Right"): neural_method(XL, modelFO_new),
        ("Foot Off", "Left"): neural_method(XR, modelHS_new),
        ("Foot Off", "Right"): neural_method(XL, modelHS_new),

        # TODO: Try re-matching these...
        #   'R' to 'Right', 'FO' to 'Foot Off', etc
        #   No, it needs to be the other way...
        # ("Foot Strike", "Left"): neural_method(XL, modelHS_new),
        # ("Foot Strike", "Right"): neural_method(XR, modelHS_new),
        # ("Foot Off", "Left"): neural_method(XL, modelFO_new),
        # ("Foot Off", "Right"): neural_method(XR, modelFO_new),
    }

    a_file = open(filename_out, 'w')
    writer = csv.writer(a_file)
    for key, value in events.items():
        writer.writerow([key, value])

    a_file.close()

    return
