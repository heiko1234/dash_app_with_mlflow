# basisapp

import os
import dash
import json
import mlflow
import pickle
import copy

import pandas as pd

from dash import Input, Output, State, callback_context
from dash import html
import dash_daq as daq
from dash import dcc


from pathlib import Path, PurePosixPath
# from azure.storage.blob import BlobServiceClient


# import mlflow.sklearn
import mlflow



from dotenv import load_dotenv



load_dotenv()


local_run = os.getenv("LOCAL_RUN", True)

print(f"Local run: {local_run}")

# Azure Strings and infos
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("LOCAL_BLOB_MODEL_CONTAINER_NAME", "model-container")
container_name
connection_string




# Initalise the app
app = dash.Dash(__name__, suppress_callback_exceptions = True)


# mlflow dtypes
mlflow_dtypes = {
    "float": "float32",
    "integer": "int32",
    "boolean": "bool",
    "double": "double",
    "string": "object",
    "binary": "binary",
}



# Dash app styles
Sidebar_Style = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "27rem", #16 to 20
    "padding": "2rem 1rem",
    "background-color": "grey",
}

Content_Style = {
    "margin-left": "29rem",  #18
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "white",
    "color": "black",
}


# Style shadows
box_shadow_style_dict = {
    "1": "rgba(0, 0, 0, 0.04) 0px 3px 5px",
    "2": "rgba(100, 100, 111, 0.2) 0px 7px 29px 0px",
    "3": "rgba(0, 0, 0, 0.35) 0px 5px 15px",
    "4": "rgba(0, 0, 0, 0.16) 0px 1px 4px, rgb(51, 51, 51) 0px 0px 0px 3px",
    "5": "rgba(6, 24, 44, 0.4) 0px 0px 0px 2px, rgba(6, 24, 44, 0.65) 0px 4px 6px -1px, rgba(255, 255, 255, 0.08) 0px 1px 0px inset",
    "6": "rgb(85, 91, 255) 0px 0px 0px 3px",
    "7": "rgba(255, 255, 255, 0.2) 0px 0px 0px 1px inset, rgba(0, 0, 0, 0.9) 0px 0px 0px 1px",
    "8": "rgba(99, 99, 99, 0.2) 0px 2px 8px 0px",
}



# model_dropdown_options
model_dropdown_options = [{"label": "linear", "value": "linear"},
                            {"label": "adaboost", "value": "adaboost"}]




def get_mlflow_model(model_name, azure=True, model_dir = "/model/"):

    if azure:
        model_dir = os.getenv("MLFLOW_MODEL_DIRECTORY", "models:/")
        model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Staging")
        artifact_path = PurePosixPath(model_dir).joinpath(model_name, model_stage)
        artifact_path

        model = mlflow.pyfunc.load_model(str(artifact_path))
        print(f"Model {model_name} loaden from Azure: {artifact_path}")

    if not azure:
        model = pickle.load(open(f"{model_dir}/{model_name}/model.pkl", 'rb'))
        print(f"Model {model_name} loaded from local pickle file")

    return model


def get_model_json_artifact(
    local=True,
    path=None,
    model_path="models",
    model_name=None,
    features="feature_dtypes.json",
):
    """This function loads json file form a dumped mlflow model
    Args:
        local (bool, optional): [description]. Defaults to True.
        path ([type], optional): in docker: "/model/", else folder where models are saved.
        model_path (str, optional): [description]. Defaults to "models".
        model_name ([type], optional): [sklearn model name]. Defaults to None.
        features (str, optional): feature_dtypes.json/ feature_limits.json
    Returns:
        [type]: [json file]
    """

    if local:
        if path is None:
            path = Path(__file___).parent
            # print(f"Parentspath: {path}")
    if not local:
        # Access the artifacts to "/model/model_name/file" for the docker.
        path = "/model/"
        model_path = ""

    path_load = os.path.join(path, model_path, model_name, features)

    return json.loads(open(path_load, "r").read())


def create_all_model_json_dict(local=True,
    path=None,
    model_path="models",
    features="feature_dtypes.json"):
    output = {}
    folderpath = os.path.join(path, model_path)
    for folder in os.listdir(folderpath):
        if os.path.isdir(os.path.join(folderpath, folder)):
            output[folder] = get_model_json_artifact(
                            local=local,
                            path=path,
                            model_path=model_path,
                            model_name=folder,
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


def transform_df_to_mlflow_df(data, dtype_dict, mlflow_dtypes):
    for element in list(dtype_dict.keys()):
        try:
            data[element] = data[element].astype(
                mlflow_dtypes[dtype_dict[element]]
            )
        except BaseException:
            continue
    return data



# Cards and app design

def create_card(content, id, title = "Title",box_shadow= "rgba(97, 97, 97, 0.2) 0px 2px 8px 0px", height="250px", width="250px"):

    return html.Div(id = id,
                    children=[title, content],
                    style={
                        "box-shadow": box_shadow,
                        "padding": "10px",
                        "height": height,
                        "width": width,
                        "display": "inline-block"
                    },
                )

 


def detail_card(
    id,
    height,
    width,
    title = None,
    content = None,
    box_shadow_style=box_shadow_style_dict["8"],
):

    header = html.Div(children=[
                        html.Div(
                            title,
                            style={
                                "font-size": 15,
                                "font-weight": "bold",
                                "color": "black",
                            },
                            ),
                        ],
                        style={"display": "inline-block"}
    )

    card = html.Div(
        id=id,
        children=[header, html.H3(""), content],
        style={
            "box-shadow": box_shadow_style,
            "padding": "10px",
            "height": height,
            "width": width,
            "display": "inline-block"
        },
    )

    return card



def content_card(
    id,
    title,
    content,
    height,
    width,
    box_shadow_style=box_shadow_style_dict["8"],
):

    header = html.Div(children=[
                        html.Div(
                            title,
                            style={
                                "font-size": 15,
                                "font-weight": "bold",
                                "color": "black",
                            },
                            ),
                        ],
                        style={"display": "inline-block"}
    )

    card = html.Div(
        id=id,
        children=[header, html.H3(""), *content],
        style={
            "box-shadow": box_shadow_style,
            "padding": "10px",
            "height": height,
            "width": width,
        },
    )

    return card


def gauge_color(value, min = 20, max= 60, ranges=[0,30,40,50,60]):
    return html.Div(
        daq.Gauge(
            id="gauge_id",
            color={
                "gradient": True,
                "ranges": {
                    "green": [ranges[0], ranges[1]],
                    "yellow": [ranges[1], ranges[2]],
                    "red": [ranges[2], ranges[3]],
                    "purple": [ranges[3],ranges[4]],
                },
            },
            showCurrentValue=True,
            scale={"start": 0, "interval": 1, "labelInterval": 10},
            units="Unit",
            value=value,
            label="Parameter",
            max=max,
            min=min,
            size=500,
        )
    )




page_controls = html.Div(
    id="pagecontrol",   
    children=[
        html.H3(""),
        html.Div(children=[
            html.Button(
                "User Guide",
                id = "userguide_id",
                n_clicks=0),
            html.Button(
                "Close User Guide",
                id = "close_userguide_id",
                n_clicks=0),
        ], style={"display": "flex"},
        ),
        html.H3(""),
        html.H3("Load Content?"),
        html.Button(
            "Yes/No",
            id = "button_id",
            n_clicks=0),
        html.H3(""),
        dcc.Dropdown(
            id = "model_dropdown", 
            options=model_dropdown_options,
            value=model_dropdown_options[0]["value"],
            style={"width": "100px"}
        ),
        html.H3(""),
        daq.NumericInput(
            id="MP09", 
            label="ManufacturingProcess09",
            value = 40,
            min = 0,
            max = 100,
            size = "150px"),
        html.H3(""),
        daq.NumericInput(
            id="MP13", 
            label="ManufacturingProcess13",
            value = 35,
            min = 0,
            max = 100,
            size = "150px"),
        html.H3(""),
        daq.NumericInput(
            id="MP20", 
            label="ManufacturingProcess20",
            value = 4400,
            min = 4000,
            max = 5000,
            size = "150px"),
        html.H3(""),
        daq.NumericInput(
            id="MP22", 
            label="ManufacturingProcess22",
            value = 10,
            min = 10,
            max = 20,
            size = "150px"),
        daq.NumericInput(
            id="MP32", 
            label="ManufacturingProcess32",
            value = 150,
            min= 100,
            max = 300,
            size = "150px"),
        html.H3(""),
        daq.NumericInput(
            id="BM02", 
            label="BiologicalMaterial02",
            value = 55,
            min = 0,
            max = 100,
            size = "150px"),
        html.H3(""),
    ]
)


# Main panel
main_panel_layout = html.Div(
    id="main_panel",
    children=[
        html.Div(
            children=[
                html.Div(children=[html.Div(id="main_page")]),
            ]
        ),
    ],
    style=Content_Style,
)


sidebar_layout = html.Div(
    id="panel-side",
    children=[

        html.Div(children=
            [html.Div(page_controls)]
            )
    ],
    style=Sidebar_Style,
)


# main content
@app.callback(
    Output("main_page", "children"),
    [
        Input("button_id", "n_clicks"),
    ],
)
def render_main_content(n_clicks):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    #
    if 'button_id' in changed_id:
        return([
            html.H2("Page 1"),
            html.Div(
                children=[
                    html.H3("Any Main Content"),
                    html.Div(children=[
                        html.Div(detail_card(content = dcc.Loading(id="outgauge"), id="Card_id2", height="90%", width="90%")),
                        html.Div(detail_card(content = dcc.Loading(id="outgauge2"), id="Card_id3", height="90%", width="90%")),
                        html.Div(detail_card(content= html.Div(id="userguide_content_output"), id="Card_userguide", height="90%", width="300px")),
                    ], 
                    style={"display": "flex", "height": "50%", "width": "90%", "marginLeft": "50px", "marginRight": "50px", "marginTop": "50px", "marginBottom": "50px"}
                    ),
                    html.Div(children=[
                        html.Div(detail_card(id = "warning_card", title = "Warning Card", content = html.Div(id="warning_content", style={"overflow": "auto", "height": "90px"}), height="120px", width="1500px")),
                        html.H3(""),
                    ],
                    style = {"display": "flex", "marginLeft": "50px", "marginRight": "50px", "marginTop": "50px", "marginBottom": "50px"})
                ],
                style={"display": "block", "marginLeft": "50px", "marginRight": "50px", "marginTop": "50px", "marginBottom": "50px"}
            ),
    ])

    else:
        return ([
            html.H2("404: page not defined"),
        ])




@app.callback(
    Output("outgauge", "children"),
    [
        Input("MP09", "value"),
        Input("MP13", "value"),
        Input("MP20", "value"),
        Input("MP22", "value"),
        Input("MP32", "value"),
        Input("BM02", "value"),
        Input("model_dropdown", "value")
    ]
)
# MP09, MP32, MP13, MP20, MP22, BM02
def update_gauche_value2(MPO9, MP13, MP20, MP22, MP32, BM02, model_choise):

    if model_choise == "linear":

        model = get_mlflow_model(model_name="dashapp_model", azure=True, model_dir = "/model/")

    else:

        model = get_mlflow_model(model_name="dashapp_model2", azure=True, model_dir = "/model/")


    # MPO9, MP13, MP20, MP22, MP32, BM02 = 40, 160, 36, 60, 10, 40

    data = pd.DataFrame(data= [[MPO9, MP13, MP20, MP22, MP32, BM02]], 
                    columns = ["ManufacturingProcess09",
                            "ManufacturingProcess13",
                            "ManufacturingProcess20",
                            "ManufacturingProcess22",
                            "ManufacturingProcess32",
                            "BiologicalMaterial02"]
                        )
    
    path_app = Path(__file__).parents[1]
    feature_dtype_dict = create_all_model_json_dict(local=True,
        path=path_app,
        model_path="models",
        features="feature_dtypes.json"
        )
    feature_dtype_dict = flatten_dict(nested_dict=feature_dtype_dict)


    for element in list(feature_dtype_dict.keys()):
        data[element] = data[element].astype(mlflow_dtypes[feature_dtype_dict[element]])


    value = model.predict(data)[-1]

    return gauge_color(value, min=20, max= 60, ranges=[20,30,40,50,60])



@app.callback(
    Output("outgauge2", "children"),
    [
        Input("MP09", "value"),
        Input("MP13", "value"),
        Input("MP20", "value"),
        Input("MP22", "value"),
        Input("MP32", "value"),
        Input("BM02", "value"),
    ]
)
# MP09, MP32, MP13, MP20, MP22, BM02
def update_gauche_value2(MPO9, MP13, MP20, MP22, MP32, BM02):


    model = get_mlflow_model(model_name="dashapp_model", azure=True, model_dir = "/model/")


    # MPO9, MP13, MP20, MP22, MP32, BM02 = 40, 160, 36, 60, 10, 40

    data = pd.DataFrame(data= [[MPO9, MP13, MP20, MP22, MP32, BM02]], 
                    columns = ["ManufacturingProcess09",
                            "ManufacturingProcess13",
                            "ManufacturingProcess20",
                            "ManufacturingProcess22",
                            "ManufacturingProcess32",
                            "BiologicalMaterial02"]
                        )
    
    path_app = Path(__file__).parents[1]
    feature_dtype_dict = create_all_model_json_dict(local=True,
        path=path_app,
        model_path="models",
        features="feature_dtypes.json"
        )
    feature_dtype_dict = flatten_dict(nested_dict=feature_dtype_dict)


    for element in list(feature_dtype_dict.keys()):
        data[element] = data[element].astype(mlflow_dtypes[feature_dtype_dict[element]])



    value = model.predict(data)[-1]

    return gauge_color(value, min=20, max= 60, ranges=[20,30,40,50,60])


@app.callback(
    Output("userguide_content_output", "children"),
    [
        Input("userguide_id", "n_clicks"),
        Input("close_userguide_id", "n_clicks"),
    ]
)
def userguide(n_clicks, n_clicks_close):

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]

    if "close_userguide_id" in changed_id:
        return html.Div(children=
            [dcc.Markdown("")],
            style={"padding": "5px",
                "width": "300px",
                "height": "500px",
                "overflow": "auto"})

    elif 'userguide_id' in changed_id:

        path_to_folder = Path(__file__).parents[0]
        path_to_userguide= path_to_folder.joinpath("userguide.md")

        with open(path_to_userguide, "r") as f:
            userguide_content = f.read()

        return html.Div(children=
            [dcc.Markdown(userguide_content)],
                style={"padding": "5px",
                "width": "300px",
                "height": "500px",
                "overflow": "auto"})






@app.callback(
    Output("warning_content", "children"),
    [
        Input("MP09", "value"),
        Input("MP13", "value"),
        Input("MP20", "value"),
        Input("MP22", "value"),
        Input("MP32", "value"),
        Input("BM02", "value"),
    ]
)
def update_gauche_value2(MPO9, MP13, MP20, MP22, MP32, BM02):

    path_app = Path(__file__).parents[1]

    feature_limits_dict = create_all_model_json_dict(local=True,
        path=path_app,
        model_path="models",
        features="feature_limits.json")
    feature_limits_dict


    TAG_limit_dict = flatten_consolidate_dict(nested_dict = feature_limits_dict, take_lower_min=True, take_higher_max=True)
    TAG_limit_dict

    # MPO9 = 36

    output = []
    output.append(create_warning(TAG_limit_dict=TAG_limit_dict, key="ManufacturingProcess09", value=MPO9, digits=2))
    output.append(create_warning(TAG_limit_dict=TAG_limit_dict, key="ManufacturingProcess13", value=MP13, digits=2))
    output.append(create_warning(TAG_limit_dict=TAG_limit_dict, key="ManufacturingProcess20", value=MP20, digits=2))
    output.append(create_warning(TAG_limit_dict=TAG_limit_dict, key="ManufacturingProcess22", value=MP22, digits=2))
    output.append(create_warning(TAG_limit_dict=TAG_limit_dict, key="ManufacturingProcess32", value=MP32, digits=2))
    output.append(create_warning(TAG_limit_dict=TAG_limit_dict, key="BiologicalMaterial02", value=BM02, digits=2))
    # print(output)
    
    return html.Div(children = output, style = {"display": "block"})



