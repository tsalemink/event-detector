from eventdetector.utils import *
from urllib.request import urlretrieve

import numpy as np

import sys
from ezc3d import c3d # Use ezc3d instead of btk
import os
import csv
import keras
from keras.models import load_model
keras.losses.weighted_binary_crossentropy = weighted_binary_crossentropy

def derivative(traj, nframes):
    traj_der = traj[1:nframes,:] - traj[0:(nframes-1),:]
    return np.append(traj_der, [[0,0,0]], axis=0)

    print("Trying %s" % (filename_in))
def extract_kinematics(filename_in):
    
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
             print("Error while reading marker data: %d, %s" % (i,v))
             return

        traj[i][:,0] = traj[i][:,0] #* incrementX
        traj[len(markers) + i][:,0] = traj[len(markers) + i][:,0] #* incrementX
        traj[i][:,2] = traj[i][:,2] #* incrementX
        traj[len(markers) + i][:,2] = traj[len(markers) + i][:,2] #* incrementX

    for i in range(len(markers)*2):
        traj[len(markers)*2 + i] = derivative(traj[i], nframes) 
        
    midASI = midASI #* incrementX

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
        traj_der = traj[1:nframes,:] - traj[0:(nframes-1),:]
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
    X = np.zeros( (data.shape[0], 15 + 15 + 6) )
    X[:,0:15] = data[:,0:15]
    X[:,15:30] = derivative(data[:,15:30])
    X[:,30:33] = derivative(data[:,24:27])
    X[:,33:36] = derivative(derivative(data[:,24:27]))

    Y = None
    if data.shape[1] != 32:
        Y = data[:,30:32]

    return X, Y

def neural_method(inputs, model):
    cols = list(range(15)) + [15 + i for i in range(13)] + [30 + i for i in range(6)]
    res = model.predict(inputs[:,cols].reshape((1,inputs.shape[0],len(cols))))
    peakind = peakdet(res[0], 0.7)
    frames = list(map(int, [k for k, v in peakind[0]]))
    return frames

def get_models():
    if not os.path.exists("models/FO.h5"):
        print ("Model not found. Downloading...")
        try:
            os.makedirs("models")
        except:
            pass
        model_path = "https://s3-eu-west-1.amazonaws.com/kidzinski/event-detector/FO.h5"
        urlretrieve(model_path, "models/FO.h5")
        model_path = "https://s3-eu-west-1.amazonaws.com/kidzinski/event-detector/HS.h5"
        urlretrieve(model_path, "models/HS.h5")
        print ("Model downloaded!")

get_models()
modelFO = load_model("models/FO.h5")
modelHS = load_model("models/HS.h5")

def process(filename_in, filename_out):
    idxL = [(int(i / 3)) * 3 + i for i in range(30)]
    idxR = [3 + (int(i / 3)) * 3 + i for i in range(30)]

    inputs = extract_kinematics(filename_in)
    inputsL = inputs[:, idxL]
    inputsR = inputs[:, idxR]
    XL, YL = convert_data(inputsL)
    XR, YR = convert_data(inputsR)

    events = {}
    events[("Foot Strike","Left")] = neural_method(XR, modelFO)
    events[("Foot Strike","Right")] = neural_method(XL, modelFO)
    events[("Foot Off","Left")] = neural_method(XR, modelHS)
    events[("Foot Off","Right")] = neural_method(XL, modelHS)

    a_file = open(filename_out, 'w')
    writer = csv.writer(a_file)
    for key, value in events.items():
        writer.writerow([key, value])

    a_file.close()

    return

