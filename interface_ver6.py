#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:12:54 2024

@author: annika
"""

# gennemgå hvilke vi rent faktisk endte med at bruge
import base64
import random
import os
import json
import dash
import time
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, callback
from dash.dependencies import ALL
import plotly.graph_objects as go
import plotly.express as px
from dash_holoniq_wordcloud import DashWordcloud
import dash_leaflet as dl
from dash.exceptions import PreventUpdate
import logging
import time
from dash.dependencies import Input, Output, MATCH



# loading files
image_directory = '/Users/thea/Desktop/Speciale/Holocaust_images'  
csv_file = '/Users/thea/Desktop/Speciale/detected_objects_full.csv'  
df_metadata = pd.read_csv('/Users/thea/Desktop/Speciale/full_metadata_geocoded.csv')



# bounding box data - might not need
df_bb = pd.read_csv(csv_file)
df_bb['Photograph Number'] = df_bb['Filename']
df_sorted = df_bb.sort_values(['Photograph Number', 'Score'], ascending=[True, False])
df_top5 = df_sorted.groupby('Photograph Number').head(10)

# map data
df_map = pd.read_csv('/Users/thea/Desktop/Speciale/full_metadata_geocoded.csv')  
df_map['latitude'] = pd.to_numeric(df_map['latitude'], errors='coerce')
df_map['longitude'] = pd.to_numeric(df_map['longitude'], errors='coerce')
df_map_droprows = df_map.dropna(subset=['latitude', 'longitude'])
df_map_droprows['Image Count'] = df_map_droprows.groupby('Location')['Location'].transform('count')

# index reset, FOR THE MAP LOCATION TO BE SHOWED CORRECTLY DONT REMOVE :C
df_map_droprows = df_map_droprows.reset_index(drop=True)

# color mapping
location_colors = {
    "Bergen-Belsen, Germany": "rgb(132, 108, 91)",
    "Theresienstadt, Czech Republic": "rgb(132, 108, 91)",
    "Buchenwald, Germany": "rgb(132, 108, 91)",
    "Gurs, France": "rgb(132, 108, 91)",
    "Sachsenhausen, Germany": "rgb(132, 108, 91)",
    "Dachau, Germany": "rgb(132, 108, 91)",
    "Beaune-la-Rolande, France": "rgb(132, 108, 91)",
    "Auschwitz, Poland": "rgb(132, 108, 91)",
    "Westerbork, The Netherlands": "rgb(132, 108, 91)",
    "Camp Pithiviers, France": "rgb(132, 108, 91)",
    "Lodz, Poland": "rgb(162, 149, 135)"
}

# default color
default_color = "#333333"

# size of the markers
def get_radius(image_count):
    base_radius = 5
    max_radius = 20  
    scale_factor = 0.5  
    radius = base_radius + scale_factor * image_count
    return min(radius, max_radius)

# converting df rows into a list of circle markers with tooltips
markers = [
    dl.CircleMarker(
        id={'type': 'marker', 'index': index},
        center=[row['latitude'], row['longitude']],
        radius=get_radius(row['Image Count']),
        children=dl.Tooltip(f"{row['Location']} ({row['Image Count']} images)"),
        color=location_colors.get(row['Location'], default_color),
        fillColor=location_colors.get(row['Location'], default_color),
        fillOpacity=0.6,
        interactive=True,
    )
    for index, row in df_map_droprows.iterrows()
]

# legend for the map - make box smaller, circles bigger
legend = html.Div(
    children=[
        #html.H4("Legend"),
        html.Ul([
            html.Li("Camps", style={'color': 'rgb(32, 108, 91)', 'margin-top': '4px'}),
            html.Li("Ghetto", style={'color': 'rgb(162, 149, 135)', 'margin-top': '4px'}),
            html.Li("Others", style={'color': '#333333', 'margin-top': '4px'})
        ])
    ],
    style={'position': 'absolute', 'bottom': '5px', 'left': '2px', 'z-index': '1000', 'background': 'white', 'padding': '5px', 'border-radius': '2px'}
)

#  Dash app
app = Dash(__name__)

######### FUNCTIONS ###########

#### ENCODING

def filter_valid_images(image_filenames):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    return [img for img in image_filenames if img.lower().endswith(valid_extensions)]

image_filenames = filter_valid_images(os.listdir(image_directory))

def encode_image_for_plotly(image_filename):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    if not image_filename.lower().endswith(valid_extensions):
        found = False
        for ext in valid_extensions:
            modified_filename = f"{image_filename}{ext}"
            image_path = os.path.join(image_directory, modified_filename)
            if os.path.isfile(image_path):
                image_filename = modified_filename
                found = True
                break
        if not found:
            print(f"No valid image file found for base filename: {image_filename}")
            return None
    
    else:
        image_path = os.path.join(image_directory, image_filename)
    mime_types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png'}
    file_ext = os.path.splitext(image_filename.lower())[1]
    mime_type = mime_types.get(file_ext, 'application/octet-stream')

    try:
        with open(image_path, 'rb') as img:
            encoded = base64.b64encode(img.read()).decode('ascii')
        return f'data:{mime_type};base64,{encoded}'
    except FileNotFoundError:
        print(f"File not found: {image_path}")
    except IOError as e:
        print(f"Error reading file {image_filename}: {e}")

    return None

############## IMAGE DISPLAY 

detected_image_directory = '/Users/thea/Desktop/Speciale/detected_images_new_new'

def create_figure(image_filename, with_bboxes=False):
    extension = '.jpg'  # nok her problemet med png stammer fra ?
    if with_bboxes:
        image_filename = f"{image_filename}_detect{extension}"  
        image_path = os.path.join(detected_image_directory, image_filename)
    else:
        image_filename = f"{image_filename}{extension}"  
        image_path = os.path.join(image_directory, image_filename)

    encoded_image = encode_image_for_plotly(image_path)
    if not encoded_image:
        print("Failed to encode image:", image_path)  
        return go.Figure()

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=encoded_image,
            xref="x",
            yref="y",
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            opacity=1,
            layer="below"
        )
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_xaxes(showticklabels=False, range=[0, 1], showgrid=False)
    fig.update_yaxes(showticklabels=False, range=[0, 1], scaleanchor="x", showgrid=False)

    return fig

################## WORDCLOUD SECTION 
df_objects = pd.read_csv('/Users/thea/Desktop/Speciale/detected_objects_full.csv')


def create_object_list(df):
    df_objects['Detected Objects'] = df['Detected Objects'].astype(str)
    object_counts = {}
    object_column = df['Detected Objects']

    for string in object_column:
        objects = string.split(', ')
        for obj in objects:
            object_counts[obj] = object_counts.get(obj, 0) + 1

    object_list = [[obj, count] for obj, count in object_counts.items()]
    object_list = sorted(object_list, key=lambda x: x[1], reverse=True)

    return object_list


# FUNCTION TO NORMALISE WORD CLOUD DATA (COPY-PASTED FROM MODULE DOCUMENTATION)
def normalise(lst, vmax=75, vmin=24):
    lmax = max(lst, key=lambda x: x[1])[1]
    lmin = min(lst, key=lambda x: x[1])[1]
    vrange = vmax-vmin
    lrange = lmax-lmin or 1
    for entry in lst:
        entry[1] = int(((entry[1] - lmin) / lrange) * vrange + vmin)
    return lst

# CALL CREATE_OBJECT_LIST() AND NORMALISE() FUNTIONS ON DATAFRAME TO CREATE OBJECT LIST
object_list = normalise(create_object_list(df_objects))
object_counts = df_objects['Detected Objects'].str.split(', ').explode().value_counts().to_dict()

def create_word_cloud_data(df, detected_objects_in_image=None):
    
    if detected_objects_in_image is None:
        detected_objects_in_image = []

    # Normalize and prepare data for the word cloud
    word_cloud_data = [
        {'text': obj, 'size': count}
        for obj, count in object_counts.items()
    ]

    return word_cloud_data



################## GALLERY FUNCTION SECTION 
max_images_to_display = 12  

def create_gallery(image_filenames): # look into changing the thumbnails in the gallery to not be squares - be original dimensions
    return html.Div([
        html.Img(
            src=encode_image_for_plotly(os.path.join(image_directory, img)),
            id={'type': 'dynamic-img', 'filename': img},  
            style={'width': '100px', 'height': '100px', 'padding': '5px', 'cursor': 'pointer'},
            n_clicks=0
        ) for img in image_filenames if encode_image_for_plotly(img) is not None
    ], style={'width': '70%', 'height': '50vh', 'overflowY': 'auto', 'flex': '1', 'padding-left': '35px', 'padding-top': '20px'})

initial_images = random.sample(image_filenames, min(len(image_filenames), max_images_to_display))
initial_gallery = create_gallery(initial_images)

################## SIDEBAR FILTERING 
def create_dropdown_options(column_name):
    return [{'label': i, 'value': i} for i in df_metadata[column_name].dropna().unique()]

def create_detected_objects_options():
    detected_objects = df_bb['Detected Objects'].unique()
    return [{'label': obj, 'value': obj} for obj in detected_objects]

# unique years 
unique_years = set()
df_metadata['Year'].dropna().apply(lambda x: unique_years.update(x.split(', ')))
unique_years = sorted(unique_years)

if 'Unknown' in df_metadata['Year'].values:
    unique_years.append('Unknown')

def create_year_options(years):
    return [{'label': year, 'value': year} for year in years]

year_options = create_year_options(unique_years)


def initialize_gallery_and_big_image():
    if not image_filenames:
        return None, "Image not available"  

    default_images = random.sample(image_filenames, min(len(image_filenames), max_images_to_display))

    gallery_children = create_gallery(default_images)

    initial_image_filename = default_images[0] if default_images else None

    if initial_image_filename:
        fig = create_figure(initial_image_filename)
        big_image_display_children = dcc.Graph(figure=fig) if fig else "Image not available"
    else:
        big_image_display_children = "Image not available"

    return gallery_children, big_image_display_children, fig
gallery_children, big_image_display_children, fig = initialize_gallery_and_big_image()

########### LAYOUT #################
app.layout = html.Div([
    # sidebar
    html.Div([
        html.H2('Filters', style={'color': 'white', 'margin-bottom': '20px', 'font-family': 'helvetica neue'}),
        html.Label('Artist', style={'color': 'white', 'font-family': 'helvetica neue'}),
        dcc.Dropdown(
            id='filter-artist',
            options=create_dropdown_options('Artist'),
            multi=True,
            style={'margin-bottom': '10px'}
        ),
        html.Label('Location', style={'color': 'white', 'font-family': 'helvetica neue'}),
        dcc.Dropdown(
            id='filter-location',
            options=create_dropdown_options('Location'),
            multi=True,
            style={'margin-bottom': '10px'}
        ),
        html.Label('Material', style={'color': 'white', 'font-family': 'helvetica neue'}),
        dcc.Dropdown(
            id='filter-material',
            options=create_dropdown_options('Material'),
            multi=True,
            style={'margin-bottom': '10px'}
        ),
        html.Label('Technique', style={'color': 'white', 'font-family': 'helvetica neue'}),
        dcc.Dropdown(
            id='filter-technique',
            options=create_dropdown_options('Technique'),
            multi=True,
            style={'margin-bottom': '10px'}
        ),
        html.Label('Database', style={'color': 'white', 'font-family': 'helvetica neue'}),
        dcc.Dropdown(
            id='filter-database',
            options=create_dropdown_options('Database'),
            multi=True,
            style={'margin-bottom': '10px'}
        ),
        
        html.Label('Detected Objects', style={'color': 'white', 'font-family': 'helvetica neue'}),
        dcc.Dropdown(
            id='filter-detected-objects',
            options=create_detected_objects_options(),
            multi=True,
            style={'margin-bottom': '10px'}
        ),
        html.Label('Camp', style={'color': 'white', 'font-family': 'helvetica neue'}),
        dcc.Dropdown(
            id='filter-camp',
            options=create_dropdown_options('Camp'),
            multi=True,
            style={'margin-bottom': '10px'}
        ),
        html.Label('Ghetto', style={'color': 'white','font-family': 'helvetica neue'}),
        dcc.Dropdown(
            id='filter-ghetto',
            options=create_dropdown_options('Ghetto'),
            multi=True,
            style={'margin-bottom': '10px'}
        ),
        html.Label('Year', style={'color': 'white','font-family': 'helvetica neue'}),
        dcc.Dropdown(
            id='filter-year',
            options=year_options,
            multi=True,  
            placeholder="Select a year",
            style={'width': '100%'}
        ),

       
        html.Button('Reset Filters', id='reset-filters-btn', n_clicks=0, style={'margin-top': '20px', 'width': '100%', 'font-family': 'helvetica neue'})
        

    ], style={'width': '15%', 'height': '100vh', 'float': 'left', 'padding': '20px', 'background-color': '#333333'}),
    # Gallery and Big Image Display Area
    html.Div([
        # Titles and Gallery
        html.Div([
            html.Div([
                html.H2('Gallery', style={'color': 'black', 'margin-bottom': '10px', 'text-align': 'center', 'font-family': 'helvetica neue'}),
            ], style={'flex': '1', 'margin-bottom': '10px', 'overflow-y': 'auto'}),
            

            html.Div([
                html.H2(id='big-image-title', children='Big Image', style={'color': 'black', 'margin-bottom': '10px', 'text-align': 'center', 'font-family': 'helvetica neue'}),
                dcc.Store(id='bbox-toggle', data={'show_bbox': False}),
                html.Button('Show Detected Objects', id='toggle-bbox-button', n_clicks=0)
            ], style={'flex': '2', 'margin-bottom': '10px', 'text-align': 'right'}),
        ], style={'display': 'flex', 'width': '100%'}),
        html.Div(id='clicked-info', style={'color': 'black', 'margin-top': '10px', 'margin-left': '30px', 'font-size': '16px', 'font-family': 'helvetica neue'}),
        
        # Gallery and Big Image Display
        html.Div([
            html.Div(id='gallery', children=gallery_children, style={'flex': '1', 'padding': '10px', 'maxWidth': '350px', 'margin-bottom': '20px'}),
            html.Div(id='big-image-display', children=big_image_display_children, style={'flex': '2', 'padding': '10px'}),
            
        ], style={'display': 'flex', 'width': '100%', 'maxHeight': '350px', 'margin': '20px'}),

        html.Div([
            html.Div([
                #html.H2('Wordcloud', style={'color': 'black', 'margin-bottom': '10px', 'text-align': 'center'}),
                html.P("This wordcloud visualizes detected objects and their frequency in the art work corpus. Click on an object to update the gallery.",
                style={'color': 'black', 'margin-bottom': '10px', 'text-align': 'center', 'font-family': 'helvetica neue'}),
                DashWordcloud(
                    id='word-cloud',
                    list=object_list,
                    width=375, height=300,
                    gridSize=2,
                    color='grey',
                    backgroundColor='#E5ECF6',
                    shuffle=False,
                    rotateRatio=0.0,
                    shrinkToFit=True,
                    hover=True
                ),
            ], style={'width': '33%', 'height': '30vh','padding': '5px'}),
            html.Div([
                html.P("This map visualizes the amount and the location of the artwork corpus  Click on a location to update the gallery.",
                style={'color': 'black', 'margin-bottom': '10px', 'text-align': 'center', 'font-family': 'helvetica neue'}),
                dl.Map(
                    children=[
                        dl.TileLayer(), 
                        dl.LayerGroup(markers), 
                        legend  
                    ],
                    style={'width': '370px', 'height': '300px', 'margin-left': '20px', 'margin-right': '20px', 'position': 'relative'}, 
                    center=(df_map_droprows['latitude'].mean(), df_map_droprows['longitude'].mean()) if not df_map_droprows.empty else [0, 0],
                    zoom=4  
                ),
            ], style={'width': '33%'}),
       
            html.Div([
                html.H2('Infobox', style={'color': 'black', 'margin-bottom': '10px', 'text-align': 'center', 'font-family': 'helvetica neue'}),
                html.P('Artist: [Artist Name]'),
                html.P('Year: [Year]'),
                html.P('Title: [Title]'),
                html.P('Location: [Location]'),
            ], id='infobox', style={'width': '33%', 'padding': '10px',  'height': '270px', 'margin-left': '20px', 'margin-right': '40px', 'overflow-y': 'auto', 'font-family': 'helvetica neue'}),
        ], style={'display': 'flex', 'width': '100%', 'padding-top': '60px', 'margin': '20px'}),
    ], style={'display': 'flex', 'flexDirection': 'column', 'flex': '1', 'padding-left': '20px'})
])



################## CALLBACK SECTION 

@app.callback(
    [Output('big-image-display', 'children'),
     Output('infobox', 'children'),
     Output('big-image-title', 'children'),
     Output('bbox-toggle', 'data')],
    [Input({'type': 'dynamic-img', 'filename': ALL}, 'n_clicks'),
     Input('toggle-bbox-button', 'n_clicks')],
    [State({'type': 'dynamic-img', 'filename': ALL}, 'id'),
     State('bbox-toggle', 'data')]
)
def update_output_and_infobox(image_clicks, toggle_click, ids, bbox_data):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id']
    print("Triggered ID:", trigger_id)  

    if 'toggle-bbox-button.n_clicks' in trigger_id:
        new_bbox_state = not bbox_data.get('show_bbox', False)
        bbox_data['show_bbox'] = new_bbox_state
        selected_image = bbox_data.get('last_selected_image', None)
        if not selected_image:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    else:
        json_string = trigger_id.split('.')[0]
        try:
            selected_image = json.loads(json_string)['filename']
            bbox_data['last_selected_image'] = selected_image
        except json.JSONDecodeError:
            return "Invalid image data", dash.no_update, dash.no_update, dash.no_update
        
        new_bbox_state = bbox_data.get('show_bbox', False)

    fig = create_figure(selected_image, with_bboxes=new_bbox_state)
    if not fig:
        return "Image not available", dash.no_update, dash.no_update, dash.no_update

    filtered_df = df_metadata[df_metadata['Photograph Number'] == selected_image].iloc[0] if not df_metadata[df_metadata['Photograph Number'] == selected_image].empty else None
    if filtered_df is None:
        return "Metadata not available", dash.no_update, dash.no_update, dash.no_update

    big_image_title = f"{filtered_df.get('Title', 'Unknown Title')} by artist {filtered_df.get('Artist', 'Unknown Artist')}"
    copyright_info = html.A(
        children=f"© Courtesy of {filtered_df.get('Database', 'Unknown')}",
        href=filtered_df.get('URL', '#'),
        target="_blank",
        style={'color': 'black', 'text-decoration': 'none', 'font-family': 'Helvetica Neue'}
    )

    infobox_content = html.Div([
        html.H2(f"Title: {filtered_df.get('Title', 'Unknown Title')}"),
        html.P(f"Artist: {filtered_df.get('Artist', 'Unknown Artist')}"),
        html.P(f"Year: {filtered_df.get('Date', 'Unknown Date')}"),
        html.P(f"Location: {filtered_df.get('Location', 'Unknown Location')}"),
        html.P(f"Camp: {filtered_df.get('Camp', 'Unknown Camp')}"),
        html.P(f"Ghetto: {filtered_df.get('Ghetto', 'Unknown Ghetto')}"),
        html.P(f"Caption: {filtered_df.get('Caption', 'Unknown Caption')}"),
        copyright_info  # Include the copyright information here.
    ])

    return [dcc.Graph(figure=fig, style={'height': '50vh'}), copyright_info], infobox_content, big_image_title, bbox_data


@app.callback(
    Output('gallery', 'children'),
    Output('clicked-info', 'children'),  
    [
        Input('filter-artist', 'value'),
        Input('filter-location', 'value'),
        Input('filter-material', 'value'),
        Input('filter-technique', 'value'),
        Input('filter-database', 'value'),
        Input('filter-detected-objects', 'value'),
        Input('filter-camp', 'value'),
        Input('filter-ghetto', 'value'),
        Input('filter-year', 'value'),
        Input({'type': 'marker', 'index': ALL}, 'n_clicks'),
        Input('word-cloud', 'click')  
    ]
)
def update_gallery(selected_artist, selected_location, selected_material, selected_technique, selected_database, selected_objects, selected_camp, selected_ghetto, selected_year, marker_clicks, word_click):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None
    location_clicked_message = ""
    word_clicked_message = ""

    if trigger_id and 'word-cloud.click' in trigger_id:

        if not word_click:
            raise PreventUpdate
        selected_word = word_click[0] if isinstance(word_click, list) and word_click else word_click
        word_clicked_message = f"You clicked on the object {selected_word}"
        filtered_df = df_bb[df_bb['Detected Objects'].str.contains(selected_word, case=False, na=False)]
        
    elif trigger_id and 'marker' in trigger_id:
        marker_id = json.loads(trigger_id.split('.')[0])
        index_clicked = marker_id['index']
        location_clicked = df_map_droprows.iloc[index_clicked]['Location']
        location_clicked_message = f"You clicked on the location {location_clicked}"
        filtered_df = df_metadata[df_metadata['Location'] == location_clicked]
    
    else:
        filtered_df = df_metadata
        if any([selected_artist, selected_location, selected_material, selected_technique, selected_database, selected_objects, selected_camp, selected_ghetto, selected_year]):
            if selected_artist:
                filtered_df = filtered_df[filtered_df['Artist'].isin(selected_artist)]
            if selected_location:
                filtered_df = filtered_df[filtered_df['Location'].isin(selected_location)]
            if selected_material:
                filtered_df = filtered_df[filtered_df['Material'].isin(selected_material)]
            if selected_technique:
                filtered_df = filtered_df[filtered_df['Technique'].isin(selected_technique)]
            if selected_database:
                filtered_df = filtered_df[filtered_df['Database'].isin(selected_database)]
            if selected_objects:
                filtered_df = df_bb[df_bb['Detected Objects'].apply(lambda x: any(obj in x for obj in selected_objects))]
            if selected_camp:
                filtered_df = filtered_df[filtered_df['Camp'].isin(selected_camp)]
            if selected_ghetto:
                filtered_df = filtered_df[filtered_df['Ghetto'].isin(selected_ghetto)]
            if selected_year:
                filtered_df = filtered_df[filtered_df['Year'].apply(lambda x: any(year in x for year in selected_year))]
            
        else:
            filtered_df = df_metadata.sample(min(max_images_to_display, len(df_metadata)))

    filtered_images = filtered_df['Photograph Number'].unique().tolist()
    clicked_info = location_clicked_message or word_clicked_message

    return create_gallery(filtered_images), clicked_info


@app.callback(
    Output('filter-artist', 'value'),
    Output('filter-location', 'value'),
    Output('filter-material', 'value'),
    Output('filter-technique', 'value'),
    Output('filter-database', 'value'),
    Output('filter-detected-objects', 'value'),
    Output('filter-camp', 'value'),
    Output('filter-ghetto', 'value'),
    Output('filter-year', 'value'),
    [Input('reset-filters-btn', 'n_clicks')]
)
def reset_filters(n_clicks):
    if n_clicks > 0:
        return None, None, None, None, None, None, None, None, None
    else:
        raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True)