

# from os import getenv
from dotenv import load_dotenv

from pathlib import PurePosixPath
from pathlib import Path
import pickle
import mlflow

import pandas as pd
# import numpy as np

# import logging
import os
import json
import copy
import mlflow
from pathlib import Path, PurePosixPath
import pickle

from azure.storage.blob import BlobServiceClient


from dash import dcc as dcc


from numpy.random import randint
from numpy.random import rand


load_dotenv()

local_run = os.getenv("LOCAL_RUN", False)
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("BLOB_MODEL_CONTAINER_NAME")



def get_mlflow_model(model_name, azure=True, local_model_dir = "/model/"):

    if azure:
        azure_model_dir = os.getenv("MLFLOW_MODEL_DIRECTORY", "models:/")
        model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Staging")
        artifact_path = PurePosixPath(azure_model_dir).joinpath(model_name, model_stage)
        artifact_path

        model = mlflow.pyfunc.load_model(str(artifact_path))
        print(f"Model {model_name} loaden from Azure: {artifact_path}")

    if not azure:
        model = pickle.load(open(f"{local_model_dir}/{model_name}/model.pkl", 'rb'))
        print(f"Model {model_name} loaded from local pickle file")

    return model


def read_model_json_from_blob(connection_string, container_name, model_name, filename):
    # get mlflow model directory in blob: "models:/""
    model_dir = os.getenv("MLFLOW_MODEL_DIRECTORY", "models:")
    # get stage: "Staging"
    model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Staging")
    # get artifact path of mode with model_name on Stage: "Staging"
    artifact_path = PurePosixPath(model_dir).joinpath(model_name, model_stage)
    # load that model
    model = mlflow.pyfunc.load_model(str(artifact_path))
    # get the loaded model runid
    model_id=model.metadata.run_id
    
    client = BlobServiceClient.from_connection_string(
        connection_string
    )
    # container blob client to container of mlflow
    container_client = client.get_container_client(container_name)

    # create file client for blob with a specific filename, of staged model

    for blob in container_client.list_blobs():
        if model_id in blob.name and filename in blob.name:
            # print(blob.name)

            f_client = client.get_blob_client(
                container=container_name, blob=blob.name
            )
    
            tempfile = os.path.join("temp.json")
            # dir_to_create = "".join(tempfile.split("/")[0:-1])
            # make folder path if it does not exist
            # os.makedirs(dir_to_create, exist_ok=True)

            with open(tempfile, "wb") as file:
                blob_data = f_client.download_blob()
                blob_data.readinto(file)

            try: 
                return json.loads(open(tempfile, "r").read())
            # except BaseException:
            #    print(f"seem to be no file: {filename} in blob: {container_name} available")
            finally:
                # finally remove temporary file
                Path(tempfile).unlink()


def get_model_json_artifact(
    azure=True,
    path=None,
    model_name=None,
    features="feature_dtypes.json",
):
    """This function loads json file form a dumped mlflow model or
    temporary dumps it to load it directly from azure / azurite

    Args:
        azure (bool, optional): [description]. Defaults to True.
        path ([type], optional): in docker: "/model/", else folder where models are saved.
        model_path (str, optional): [description]. Defaults to "models".
        model_name ([type], optional): [sklearn model name]. Defaults to None.
        features (str, optional): feature_dtypes.json/ feature_limits.json

    Returns:
        [type]: [json file]
    """

    if not azure:
        # Access the artifacts to "/model/model_name/file" for the docker.

        path_load = os.path.join(path, model_name, features)

        return json.loads(open(path_load, "r").read())
    
    if azure:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name = os.getenv("CONTAINER_NAME")

        file = read_model_json_from_blob(connection_string=connection_string, 
                        container_name=container_name, 
                        model_name=model_name, 
                        filename=features)
        if file: 
            return file
        else: 
            print(f"Warning: seem to be no file: {features} in blob: {container_name} available")


def create_all_model_json_dict(local=True,
    path=None,
    model_path=None,
    features="feature_dtypes.json",
    list_of_models=None):

    output = {}
    if local: 
        if model_path:
            folderpath = os.path.join(path, model_path)
        else:
            folderpath = path
        for folder in os.listdir(folderpath):
            if os.path.isdir(os.path.join(folderpath, folder)):
                output[folder] = get_model_json_artifact(
                                azure=False,
                                path=folderpath,
                                model_name=folder,
                                features=features,
                            )

    if not local and list_of_models:
        for modelname in list_of_models:
            output[modelname]=get_model_json_artifact(
                                    azure=True,
                                    path=None,
                                    model_name=modelname,
                                    features=features,
                                )
    return output


def flatten_dict(nested_dict):
    output={}
    for key in nested_dict.keys():
        for second_key in nested_dict[key].keys():
            if second_key not in output:
                output[second_key] = nested_dict[key][second_key]
    return output


def flatten_consolidate_dict(nested_dict, take_lower_min=True, take_higher_max=True):
    output={}
    for key in nested_dict.keys():
        for second_key in nested_dict[key].keys():
            if second_key not in output:
                output[second_key] = copy.deepcopy(nested_dict[key][second_key])
            if second_key in output:
                if take_lower_min:
                    if output[second_key]["min"] > nested_dict[key][second_key]["min"]:
                        output[second_key]["min"] = copy.deepcopy(nested_dict[key][second_key]["min"])
                else:
                    if output[second_key]["min"] < nested_dict[key][second_key]["min"]:
                        output[second_key]["min"] = copy.deepcopy(nested_dict[key][second_key]["min"])
                if take_higher_max:
                    if output[second_key]["max"] < nested_dict[key][second_key]["max"]:
                        output[second_key]["max"] = copy.deepcopy(nested_dict[key][second_key]["max"])
                else:
                    if output[second_key]["max"] > nested_dict[key][second_key]["max"]:
                        output[second_key]["max"] = copy.deepcopy(nested_dict[key][second_key]["max"])
    return output


def create_warning(TAG_limit_dict, key, value, digits=2):
    if key in TAG_limit_dict.keys():
        if value < TAG_limit_dict[key]["min"]:
            return dcc.Markdown(
                f"""{key} is below min value of model:
                        {round(TAG_limit_dict[key]["min"], digits)} """
            )

        elif value > TAG_limit_dict[key]["max"]:
            return dcc.Markdown(
                f"""{key} is above max value of model:
                        {round(TAG_limit_dict[key]["max"], digits)} """
            )

        else:
            return None

    else:
        return None


def decode_df_mlflow_dtype(data, dtype_dict):

    mlflow_dtypes = {
        "float": "float32",
        "integer": "int32",
        "boolean": "bool",
        "double": "double",
        "string": "object",
        "binary": "binary",
    }

    for element in list(dtype_dict.keys()):
        try:
            data[element] = data[element].astype(
                mlflow_dtypes[dtype_dict[element]]
            )
        except BaseException:
            continue
    return data


def create_polymer_data(M_per, Xf, SA):
    SASA = SA**2
    SASASA = SA**3
    XfXf = Xf**2
    XfXfXf = Xf**3
    return pd.DataFrame(
        data=[[M_per, Xf, SA, SASA, SASASA, XfXf, XfXfXf]],
            columns=["M%", "Xf", "SA", "SASA", "SASASA", "XfXf", "XfXfXf"],
        )





# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
    decoded = list()
    largest = 2 ** n_bits
    for i in range(len(bounds)):
        # extract the substring
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end]
        # convert bitstring to a string of chars
        chars = "".join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        ratio = integer / largest
        value = bounds[i][0] + ratio * (bounds[i][1] - bounds[i][0])
        # store
        decoded.append(value)
    return decoded


# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1) - 2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]


# lossfucntion
def lossfunction(target, X, dtype_dict, model):
    idata = create_polymer_data(M_per=X[0], Xf=X[1], SA=X[2])
    decode_df_mlflow_dtype(data=idata, dtype_dict=dtype_dict)
    modeloutput = model.predict(idata)
    return (target-modeloutput)**2


# genetic algorithm
def genetic_algorithm(
    objective=None,
    target=None,
    bounds=None,
    dtype_dict=None,
    model=None,
    break_accuracy=0.005,
    digits=5,
    n_bits=16,
    n_iter=100,
    n_pop=100,
    r_cross=0.9,
    r_mut=None,
):
    """genetic algorithm will compute on the objectiv loss function
    and given bounds for the features in the loss function a suggestion
    for new values for the model or loss function

    Args:
        objective ([function]): a loss function
        target ([number]): target value to optimize for
        bounds ([list]): a list for lower and upper limits
        break_accuracy (float): Min Difference to break,
        Defaults to 0.005.
        digits (int): number of digits for solution
        displayed Defaults to 5.
        dtype_dict ([dict]): for the objective inner function, opt
        model ([mlflow model]): for the objective inner function, opt
        n_bits (int): number of bits for a number. Defaults to 16.
        n_iter (int): number for iterations. Defaults to 100.
        n_pop (int): number of solutions test per iteration.
        Defaults to 100.
        r_cross (float): value for intercrossing. Defaults to 0.9.
        r_mut ([type]): value for mutations.
        Defaults to None: r_mut = 1.0 / (float(n_bits) * len(bounds))

    Returns:
        [type]: [description]
    """
    if r_mut is None and bounds is not None:
        r_mut = 1.0 / (float(n_bits) * len(bounds))
    else:
        r_mut = 0.5
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best = 0
    if not dtype_dict:
        best_eval = objective(target=target, X=decode(bounds, n_bits, pop[0]))
    if dtype_dict: 
        best_eval = objective(target=target, X=decode(bounds, n_bits, pop[0]), dtype_dict=dtype_dict, model=model)
    # enumerate generations
    for gen in range(n_iter):
        if best_eval <= break_accuracy:
            break
        else:
            # decode population
            decoded = [decode(bounds, n_bits, p) for p in pop]
            # evaluate all candidates in the population
            if not dtype_dict:
                scores = [objective(target=target, X=d) for d in decoded]
            if dtype_dict:
                scores = [objective(target=target, X=d, dtype_dict=dtype_dict, model=model) for d in decoded]
            # check for new best solution
            for i in range(n_pop):
                if scores[i] < best_eval:
                    best, best_eval = pop[i], scores[i]
                    # print(">%d, new best f(%s) =
                    # %f" % (gen, decoded[i], scores[i]))
                    rs = round(scores[i][0], digits)
                    print(f">{gen}, new best {decoded[i]} = {rs}")
            # select parents
            selected = [selection(pop, scores) for _ in range(n_pop)]
            # create the next generation
            children = list()
            for i in range(0, n_pop, 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i + 1]
                # crossover and mutation
                for c in crossover(p1, p2, r_cross):
                    # mutation
                    mutation(c, r_mut)
                    # store for next generation
                    children.append(c)
            # replace population
            pop = children
    print("Done!")
    # return [best, best_eval]
    decoded = decode(bounds, n_bits, best)
    rounded_values = [round(element, digits) for element in decoded]
    return rounded_values
