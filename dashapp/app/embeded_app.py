# Embedded App only for Argus Layouting

# import dash
# import dash_html_components as html
from dash import html

from app.basisapp import sidebar_layout
from app.basisapp import main_panel_layout
from app.basisapp import app

# Root
app.layout = html.Div(id="root", children=[sidebar_layout, main_panel_layout])


