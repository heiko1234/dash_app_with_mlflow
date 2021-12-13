# basisapp

import os
import dash


import pandas as pd

from dash import Input, Output, State, callback_context
from dash import html
import dash_daq as daq
from dash import dcc
from dash import dash_table

from pathlib import Path

# utility functions 
from app.app_utility import get_mlflow_model, get_model_json_artifact
from app.app_utility import decode_df_mlflow_dtype, create_warning
from app.app_utility import flatten_consolidate_dict, flatten_dict
from app.app_utility import create_polymer_data

from app.app_utility import lossfunction, genetic_algorithm


from dotenv import load_dotenv



load_dotenv()


local_run = os.getenv("LOCAL_RUN", True)

print(f"Local run: {local_run}")

# Azure Strings and infos
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("CONTAINER_NAME", "model-container")
container_name
connection_string



MFI_model= get_mlflow_model(azure=True, model_name="MFI_polymer")
CI_model= get_mlflow_model(azure=True, model_name="CI_polymer")

CI_dtype_dict=get_model_json_artifact(model_name= "CI_polymer", features="feature_dtypes.json", azure=True)
MFI_dtype_dict=get_model_json_artifact(model_name= "MFI_polymer", features="feature_dtypes.json", azure=True)

CI_limits_dict=get_model_json_artifact(model_name= "CI_polymer", features="feature_limits.json", azure=True)
MFI_limits_dict=get_model_json_artifact(model_name= "MFI_polymer", features="feature_limits.json", azure=True)



# custom funcitons
def generate_modal(markdown_text):
    return html.Div(
        id="markdown",
        className="modal",
        children=(
            html.Div(
                id="markdown-container",
                className="markdown-container",
                children=[
                    #html.Div(
                    #    className="close-container",
                    #    children=html.Button(
                    #        "Close",
                    #        id="markdown_close",
                    #        n_clicks=0,
                    #        className="closeButton",
                    #    ),
                    #),
                    html.Div(
                        className="markdown-text",
                        children=dcc.Markdown(markdown_text)
                        ),
                ],
            )
        ),
    )





# Initalise the app
app = dash.Dash(__name__, suppress_callback_exceptions = True)




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


def gauge_color(value, min = 20, max= 60, ranges=[0,30,40,50,60], color=None, label = "Parameter"):
    
    if not color:
        color ={
            "gradient": True,
            "ranges": {
                "green": [ranges[0], ranges[1]],
                "yellow": [ranges[1], ranges[2]],
                "red": [ranges[2], ranges[3]],
                "purple": [ranges[3],ranges[4]],
            },
        }
    return html.Div(
        daq.Gauge(
            id="gauge_id",
            color=color,
            showCurrentValue=True,
            units="Unit",
            value=value,
            label=label,
            max=max,
            min=min,
            size=350,
        )
    )



# page controls
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
        #html.H3("select a model"),
        #html.H3(""),
        #dcc.Dropdown(
        #    id = "model_dropdown", 
        #    options=model_dropdown_options,
        #    value=model_dropdown_options[0]["value"],
        #    style={"width": "100px"}
        #),
        html.H3(""),
        dcc.Tabs(
                id="app-tabs",
                value="manual_inputs",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="manual-tab",
                        label="Inputs",
                        value="manual_inputs",
                        className="custom-tab",
                    ),
                    dcc.Tab(
                        id="model-tab",
                        label="Model",
                        value="model_inputs",
                        className="custom-tab",
                    ),
                    dcc.Tab(
                        id="suggestion-tab",
                        label="Suggestion",
                        value="suggestion_inputs",
                        className="custom-tab",
                    ),
                    dcc.Tab(
                        id="evaluation-tab",
                        label="Evaluation",
                        value="evaluation_inputs",
                        className="custom-tab",
                    ),
                ],
        ),
        html.Div(id="rendered_sidebar")
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
        Input("app-tabs", "value")
    ],
)
def render_main_content(page):
    #
    if page == "manual_inputs":
        return(html.Div(
            children=[
            html.H3("Checking User Inputs"),
            html.Div(
                children=[
                    html.H3("User Inputs"),
                    html.Div(children=[
                        html.Div(detail_card(content = dcc.Loading(id="outgauge"), id="Card_id2", height="90%", width="90%")),
                        html.Div(detail_card(content = dcc.Loading(id="outgauge2"), id="Card_id3", height="90%", width="90%")),
                        html.Div(detail_card(content = dcc.Loading(id="outgauge3"), id="Card_id3", height="90%", width="90%")),
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
        ])),
    if page == "model_inputs":
       return(html.Div(
            children=[
            html.H3("Modelling"),
            html.Div(
                children=[
                   html.H3("User Inputs"),
                    html.Div(children=[
                        html.Div(detail_card(content = dcc.Loading(id="gaugemodelling"), id="Card_id4", height="90%", width="90%")),
                        html.Div(detail_card(content = dcc.Loading(id="gaugemodelling2"), id="Card_id5", height="90%", width="90%")),
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
    ),
    if page == "suggestion_inputs":
       return(html.Div(
            children=[
            html.H3("Set Point Suggestions"),
            html.Div(
                children=[
                   html.H3("Suggestions"),
                    html.Div(children=[
                        html.Div(detail_card(content = dcc.Loading(id="suggestion_id"), id="Card_id6", height="90%", width="90%")),
                        html.Div(detail_card(content = dcc.Loading(id="suggestion_id2"), id="Card_id7", height="90%", width="90%")),
                        ],
                        style={"display": "block", "marginLeft": "50px", "marginRight": "50px", "marginTop": "50px", "marginBottom": "50px"} 
                    )
                    ],
                ),
            ]
            )
       ),

    else:
        return ([
            html.H2("404: page not defined"),
        ])


# side content
@app.callback(
    Output("rendered_sidebar", "children"),
    [
        Input("app-tabs", "value")
    ]
)
def render_side_content(page):
    if (page == "manual_inputs") or (page == "model_inputs"):
        return  html.Div(
                    children=[
                        html.H3(""),
                        html.H3("M%"),
                        html.H3(""),
                        daq.Slider(
                            id="M_per", 
                            #label="M%",
                            value = 2,
                            min = 0,
                            max = 5,
                            step = 0.1,
                            color="blue",
                            handleLabel={"showCurrentValue": True,"label": "VALUE"},
                            #size = "150px"
                            ),
                        html.H3(""),
                        html.H3("Xf"),
                        html.H3(""),
                        daq.Slider(
                            id="Xf", 
                            #label="Xf",
                            value = 16,
                            min = 7.5,
                            max = 25,
                            step = 0.1,
                            handleLabel={"showCurrentValue": True,"label": "VALUE"},
                            color="blue",
                            #size = "150px"
                            ),
                        html.H3(""),
                        html.H3("SA"),
                        html.H3(""),
                        daq.Slider(
                            id="SA", 
                            #label="SA",
                            value = 60,
                            min = 40,
                            max = 90,
                            step = 0.1,
                            color="blue",
                            handleLabel={"showCurrentValue": True,"label": "VALUE"},
                            #size = "150px"
                            ),
                        html.H3(""),
                    ],
        )
    if page == "suggestion_inputs":
        return  html.Div(
                    children=[
                        html.H3(""),
                        html.H3("CI"),
                        html.H3(""),
                        daq.Slider(
                            id="CI_id", 
                            value = 90,
                            min = 0,
                            max = 200,
                            step = 1,
                            color="blue",
                            handleLabel={"showCurrentValue": True,"label": "VALUE"},
                            ),
                        html.H3(""),
                        html.H3("MFI"),
                        html.H3(""),
                        daq.Slider(
                            id="MFI_id", 
                            value = 196,
                            min = 190,
                            max = 200,
                            step = 0.1,
                            handleLabel={"showCurrentValue": True,"label": "VALUE"},
                            color="blue",
                            ),
                        html.H3(""),
                        html.Button('Execute', id='execute-button', n_clicks=0),
                        html.H3(""),
                        html.H3("-----"),
                        html.H3(""),
                        html.H3("M%"),
                        html.H3(""),
                        dcc.RangeSlider(
                            id="M_per_range", 
                            value = [2,3],
                            min = 0,
                            max = 5,
                            step = 0.1,
                            tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        html.H3(""),
                        html.H3("Xf"),
                        html.H3(""),
                        dcc.RangeSlider(
                            id="Xf_range", 
                            value = [14,18],
                            min = 7.5,
                            max = 25,
                            step = 0.1,
                            tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        html.H3(""),
                        html.H3("SA"),
                        html.H3(""),
                        dcc.RangeSlider(
                            id="SA_range", 
                            value = [50,70],
                            min = 40,
                            max = 90,
                            step = 0.1,
                            tooltip={"placement": "bottom", "always_visible": True}
                            ),

                    ]
        )

    else:
        return html.H3("to be created")



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
        
        # return {"display": "none"}


@app.callback(
    Output("outgauge", "children"),
    [
        Input("M_per", "value")
    ]
)
def create_gauge(M_per_value):
    return gauge_color(value=M_per_value, min = 0, max= 5, ranges=[0,1,3,4,5], label="M%")




@app.callback(
    Output("outgauge2", "children"),
    [
        Input("Xf", "value")
    ]
)
def create_gauge(Xf_value):
    return gauge_color(value=Xf_value, min = 7.5, max= 25, ranges=[7.5,12,15,18,25], label="Xf")



@app.callback(
    Output("outgauge3", "children"),
    [
        Input("SA", "value")
    ]
)
def create_gauge2(SA_value):
    return gauge_color(value=SA_value, min = 40, max= 90, ranges=[40,50,65,80,90], label="SA")



@app.callback(
    Output("gaugemodelling", "children"),
    [
        Input("M_per", "value"),
        Input("Xf", "value"),
        Input("SA", "value"), 
    ]
)
def create_gaugemodelling(M, Xf, SA):

    data = create_polymer_data(M_per=M, Xf = Xf, SA=SA)

    polymer_data_decoded = decode_df_mlflow_dtype(data = data, dtype_dict=MFI_dtype_dict)
    
    pvalue = round(MFI_model.predict(polymer_data_decoded)[0], 2)

    return gauge_color(value=pvalue, min = 190, max= 200, ranges=[190, 192, 196, 198,200], label="MFI")



@app.callback(
    Output("gaugemodelling2", "children"),
    [
        Input("M_per", "value"),
        Input("Xf", "value"),
        Input("SA", "value"), 
    ]
)
def create_gaugemodelling2(M, Xf, SA):

    data = create_polymer_data(M_per=M, Xf = Xf, SA=SA)

    polymer_data_decoded = decode_df_mlflow_dtype(data = data, dtype_dict=CI_dtype_dict)
    
    pvalue = round(CI_model.predict(polymer_data_decoded)[0], 2)

    return gauge_color(value=pvalue, min = 0, max= 200, ranges=[0,50,100,150,200], label="CI")





@app.callback(
    Output("warning_content", "children"),
    [
        Input("M_per", "value"),
        Input("Xf", "value"),
        Input("SA", "value"), 
    ]
)
def warnings_MFI(M, Xf, SA):

    nested_dict = {}
    nested_dict["MFI_polymer"] = MFI_limits_dict
    nested_dict["CI_polymer"] = CI_limits_dict

    feature_limits= flatten_consolidate_dict(nested_dict = nested_dict, take_lower_min=True, take_higher_max=True)

    return html.Div(
                children=[
                    create_warning(TAG_limit_dict=feature_limits, key = "M%", value=M),
                    create_warning(TAG_limit_dict=feature_limits, key = "Xf", value=Xf), 
                    create_warning(TAG_limit_dict=feature_limits, key = "SA", value=SA)
                ]
            )

     

@app.callback(
    Output("suggestion_id", "children"),
    [
        State("MFI_id", "value"),
        State("M_per_range", "value"),
        State("Xf_range", "value"),
        State("SA_range", "value"), 
        Input("execute-button", "n_clicks")

    ]
)
def SP_prediction(SP_MFI, M_per_range, Xf_range, SA_range, n_clicks):

    if n_clicks == 0:
        return html.Div(html.H3("Execute the calculation"))

    if n_clicks > 0:

        SP_MFI = round(SP_MFI, 2)

        # MFI_limits_dict

        nested_dict = {}
        nested_dict["MFI_model"] = MFI_limits_dict
        nested_dict["CI_mode"] = CI_limits_dict

        bounds = [[M_per_range[0], M_per_range[1]], [Xf_range[0], Xf_range[1]], [SA_range[0], SA_range[1]]]

        # consolidated_limit_dict = flatten_consolidate_dict(nested_dict=nested_dict, take_lower_min=True, take_higher_max=True)
        # list_of_features = ["M%",  "Xf", "SA"]

        # bounds = [[consolidated_limit_dict[element]["min"], consolidated_limit_dict[element]["max"]] for element in list_of_features]

        new_setpoints = genetic_algorithm(
            objective=lossfunction,
            target=SP_MFI,
            bounds=bounds,
            dtype_dict=MFI_dtype_dict,
            model=MFI_model,
            break_accuracy=0.09,
            digits=5,
            n_bits=16,
            n_iter=2,
            n_pop=100,
            r_cross=0.9,
            r_mut=None,
        )

        df = create_polymer_data(M_per=new_setpoints[0], Xf=new_setpoints[1], SA=new_setpoints[2])

        df=decode_df_mlflow_dtype(data=df, dtype_dict=MFI_dtype_dict)

        pred_MFI = round(MFI_model.predict(df)[0],2)

        dff = df.iloc[:,:3]
        dff = dff.round(2)



        return html.Div(
                    children=[
                        html.H3(""),
                        html.H3(f"Selected Model: MFI: {pred_MFI}"),
                        dash_table.DataTable(
                            id="table", 
                            columns = [{"name": i, "id": i} for i in dff.columns],
                            data= dff.to_dict(orient="records")
                        )
                    ]
                )



@app.callback(
    Output("suggestion_id2", "children"),
    [
        State("CI_id", "value"),
        State("M_per_range", "value"),
        State("Xf_range", "value"),
        State("SA_range", "value"), 
        Input("execute-button", "n_clicks")

    ]
)
def SP_prediction(SP_CI, M_per_range, Xf_range, SA_range, n_clicks):

    if n_clicks == 0:
        return html.Div(html.H3("Execute the calculation"))

    if n_clicks > 0:

        SP_CI = round(SP_CI, 2)

        nested_dict = {}
        nested_dict["MFI_model"] = MFI_limits_dict
        nested_dict["CI_mode"] = CI_limits_dict

        bounds = [[M_per_range[0], M_per_range[1]], [Xf_range[0], Xf_range[1]], [SA_range[0], SA_range[1]]]

        new_setpoints = genetic_algorithm(
            objective=lossfunction,
            target=SP_CI,
            bounds=bounds,
            dtype_dict=MFI_dtype_dict,
            model=MFI_model,
            break_accuracy=0.09,
            digits=5,
            n_bits=16,
            n_iter=2,
            n_pop=100,
            r_cross=0.9,
            r_mut=None,
        )

        df = create_polymer_data(M_per=new_setpoints[0], Xf=new_setpoints[1], SA=new_setpoints[2])

        df=decode_df_mlflow_dtype(data=df, dtype_dict=CI_dtype_dict)

        pred_CI = round(CI_model.predict(df)[0],2)

        dff = df.iloc[:,:3]
        dff = dff.round(2)


        return html.Div(
                    children=[
                        html.H3(""),
                        html.H3(f"Selected Model: CI: {pred_CI}"),
                        dash_table.DataTable(
                            id="table", 
                            columns = [{"name": i, "id": i} for i in dff.columns],
                            data= dff.to_dict(orient="records")
                        )
                    ]
                )



