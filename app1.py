import os
import base64
import subprocess
import io
from AIDetector_pytorch import Detector
from tracker import _nn_euclidean_distance
from tracker import list_txt
import imutils
import cv2
import pandas as pd
from textwrap import dedent
import dash
from dash import dcc
from dash import html
import dash_player as player
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pathlib
from datetime import date

doneProcess = False
targetLocked = False

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Lost and Found Detection Explorer"


def markdown_popup():
    return html.Div(
        id="markdown",
        className="modal",
        style={"display": "none"},
        children=(
            html.Div(
                className="markdown-container",
                children=[
                    html.Div(
                        className="close-container",
                        children=html.Button(
                            "Close",
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                        ),
                    ),
                    html.Div(
                        className="markdown-text",
                        children=[
                            dcc.Markdown(
                                children=dedent(
                                    """
                                ##### What am I looking at?

                                This app enhances visualization of objects detected using state-of-the-art Mobile Vision Neural Networks.
                                Most user generated videos are dynamic and fast-paced, which might be hard to interpret. A confidence
                                heatmap stays consistent through the video and intuitively displays the model predictions. The pie chart
                                lets you interpret how the object classes are divided, which is useful when analyzing videos with numerous
                                and differing objects.

                                """
                                )
                            )
                        ],
                    ),
                ],
            )
        ),
    )

# 设置上传文件夹
UPLOAD_DIRECTORY = "./images/origin"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

processing_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Processing Videos -- Status"), close_button=False),
        dbc.ModalBody(
            dcc.Loading(
                id="loading",
                children=[html.Div(id="processing-status",
                                   style={"position": "absolute", "top": "100px",
                                            'left': '80px', 'color': 'white',
                                            'font-weight': 'bold', 'font-size': '20px'})],
                fullscreen=False  # 隐藏关闭按钮
            )
        ),
        dbc.ModalFooter(
            dbc.Button("         Close         ", id="close-processing-modal", className="ml-auto",
                       style={"position": "absolute", "top": "180px", 'left': '80px',
                              'color': 'white', 'font-weight': 'bold'}),

        ),
    ],
    id="processing-modal",
    size='sm',
    backdrop="static",
    centered=True,
    style={
        'position': 'fixed',
        'top': '120px',
        'left': '400px',
        'width': '30%',
        'height': '35%',
        'background-color': '#fa4f56',  # 更改模态框的颜色
    },

)
processing_modal.children[0].style = {"position": "absolute", "top": "25px", 'left': '40px',
                                      'color': 'white', 'font-weight': 'bold', 'font-size': '25px'}

# Main App
app.layout = html.Div(
    children=[
        dcc.Interval(id="interval-updating-graphs", interval=1000, n_intervals=0),
        html.Div(id="top-bar", className="row"),
        html.Div(
            className="container",
            children=[
                html.Div(
                    id="left-side-column",
                    className="eight columns",
                    children=[
                        html.Img(
                            id="logo-mobile", src=app.get_asset_url("cat-logo.png")
                        ),
                        html.Div(
                            id="header-section",
                            children=[
                                html.H4("Lost and Found Detection Explorer"),
                                html.P(
                                    "You need to select footage"
                                    " ,then select the type of item you lost, then click the button, "
                                ),
                                html.P(
                                    "Wait patiently for a few minutes"
                                    " ,and the right side of the page will show the scene when the owner lost the item."
                                ),
                                html.Button(
                                    "Learn More", id="learn-more-button",
                                    n_clicks=0,
                                    style={'display': 'inline-block',
                                           'margin-right': '20px',
                                           'margin-top': '20px',
                                           'margin-bottom': '30px'}
                                ),
                                html.Button('Run backend', id='run-backend-btn',
                                            style={'background-color': 'black', 'color': 'white',
                                                   'border-style': 'groove', 'border-color': 'black'}),
                                html.Div(id='backend-output'),
                                processing_modal,  # 显示进程
                                dcc.Interval(
                                    id='interval-component',
                                    interval=3000  # 单位是毫秒
                                ),
                            ],
                        ),
                        html.Div(
                            className="video-outer-container",
                            children=html.Div(
                                className="video-container",
                                children=player.DashPlayer(
                                    id="video-display",
                                    url="https://youtu.be/M1rr89lYkfg",
                                    controls=True,
                                    playing=False,
                                    volume=1,
                                    width="100%",
                                    height="100%",
                                ),
                            ),
                        ),
                        html.Div(
                            className="control-section",
                            children=[
                                html.Div(
                                    className="control-element",
                                    children=[
                                        html.Div(children=["Footage Selection:"]),
                                        dcc.Dropdown(
                                            id="dropdown-footage-selection",
                                            options=[
                                                {
                                                    "label": "test video 1",
                                                    "value": "test1",
                                                },
                                                {
                                                    "label": "test video 2",
                                                    "value": "test2",
                                                },
                                                {
                                                    "label": "test video 3",
                                                    "value": "test3",
                                                },

                                            ],
                                            value="test3",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(id='footage-output', style={'display': 'inline-block', 'margin-top': '10px','margin-bottom': '10px',  'text-align': 'right'}),
                                html.Div(
                                    className="control-element",
                                    children=[
                                        html.Div(children=["Category of Lost Items:"]),
                                        dcc.Dropdown(
                                            id="items-dropdown",
                                            options=[
                                                {
                                                    "label": "suitcase",
                                                    "value": "suitcase",
                                                },
                                                {
                                                    "label": "backpack",
                                                    "value": "backpack",
                                                },
                                            ],
                                            value="suitcase",
                                            searchable=False,
                                            clearable=False,
                                        ),

                                    ],
                                ),
                                # 创建输出文本
                                html.Div(id='output',
                                         style={'display': 'inline-block', 'margin-top': '10px',
                                                'margin-bottom': '1px',
                                                'text-align': 'right'}),
                            ],
                        ),
                        # 添加一条虚线
                        html.Hr(style={'border-top': 'dashed 2px', 'width': '100%'}),
                        html.Div(
                            className="upload-element",
                            children=[
                                html.Div(children=["Upload Photo and Change File Name:"], style={"margin-bottom": "20px", "margin-left": "0"}),
                                dcc.Upload(
                                    id="upload-image",
                                    children=html.Div(["Drag the File Here or ", html.A("Choose File")]),
                                    style={
                                        "width": "100%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                    },
                                    # 允许上传多个文件
                                    multiple=True,
                                ),
                                html.Br(style={'height': '40px'}),  # 添加一个空行
                                # 显示上传的文件列表
                                html.Ul(id="file-list"),
                                # 文本输入框，用于输入文件名称
                                dcc.Input(
                                    id="new-filename",
                                    type="text",
                                    placeholder="Create a new file name",
                                    style={'display': 'inline-block', 'margin-top': '10px', 'margin-bottom': '10px',
                                           'text-align': 'left', 'margin-right': '20px'}
                                ),
                                # 提交按钮
                                html.Button("SUBMIT", id="submit-button", n_clicks=0,
                                            style={'background-color': 'black', 'color': 'white',
                                                   'border-style': 'groove', 'border-color': 'black'}),
                            ],
                            style={"max-width": "800px", "margin": "auto"},
                        ),
                        # 添加一条虚线
                        html.Hr(style={'border-top': 'dashed 2px', 'width': '85%'}),
                        html.Button('   Delete All Uploaded Files   ', id='delete-files-button',
                                    style={"margin-top": "10px", "margin-bottom": "30px",
                                           'background-color': 'black', 'color': 'white',
                                           'border-style': 'groove', 'border-color': 'black'}),
                        html.Div(id='delete-output')
                    ],
                ),
                html.Div(
                    id="right-side-column",
                    className="four columns",
                    children=[
                        html.Div(
                            className="img-container",
                            children=html.Img(
                                id="logo-web", src=app.get_asset_url("cat-logo.png")
                            ),
                        ),
                        html.Div(id="div-visual-mode"),
                        html.Div(id="div-detection-mode"),
                        html.H6("Here Are the Possible Owner & Belongings"),
                    ],
                ),
            ],
        ),
        markdown_popup(),
    ]
)

def save_file(name, content):
    """保存上传的文件到指定文件夹中"""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def delete_files():
    """删除指定文件夹中的所有文件"""
    for filename in os.listdir(UPLOAD_DIRECTORY):
        file_path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
@app.callback(
    Output("file-list", "children"),
    Input("submit-button", "n_clicks"),
    State("upload-image", "filename"),
    State("upload-image", "contents"),
    State("new-filename", "value"),
)
def save_uploaded_files(n_clicks, filenames, contents, new_filename):
    """保存上传的文件，并更改文件名称"""
    if n_clicks == 0 or not filenames:
        return [html.Li("Please Upload Your Files.")]
    elif n_clicks == 1:
        for filename, content in zip(filenames, contents):
            save_file(new_filename + pathlib.Path(filename).suffix, content)
        return [html.Li(f"{n_clicks} photo has been uploaded successfully")]
    else:
        # delete_files() # 删除指定文件夹中的所有文件
        # 保存上传的文件到指定文件夹中，并更改文件名称
        for filename, content in zip(filenames, contents):
            save_file(new_filename + pathlib.Path(filename).suffix, content)
        # 显示保存成功的消息
        return [html.Li(f"{n_clicks} photos have been uploaded successfully")]

@app.callback(Output('delete-output', 'children'),
              Input('delete-files-button', 'n_clicks'))
def delete_files(n_clicks):
    if n_clicks is None:
        return ''
    directory = "./images/origin"  # 指定要删除文件的目录路径
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(e)

    return 'Done！'


@app.callback(
    Output("processing-status", "children"),
    [Input("loading", "fullscreen")],
    [Input('interval-component', 'n_intervals')]
)
def update_processing_status(fullscreen, n_intervals):
    doneProcess = False
    targetLocked = False
    fullscreen = True
    if os.path.exists('doneProcess.txt'):
        with open('doneProcess.txt', 'r') as f:
            doneProcess = f.read()
    if os.path.exists('targetLocked.txt'):
        with open('targetLocked.txt', 'r') as f:
            targetLocked = f.read()
    if doneProcess == 'True':
        fullscreen = False
        return 'Processing has completed.'
    if targetLocked != 'True':
        return 'Searching Target...'
    else:
        return 'Target Locked'


@app.callback(
    Output("processing-modal", "is_open"),
    [Input("run-backend-btn", "n_clicks"), Input("close-processing-modal", "n_clicks")],
    [State("processing-modal", "is_open")]
)
def toggle_processing_modal(n_show, n_close, is_open):
    if n_show or n_close:
        return not is_open
    return is_open


def run_backend():
    # Run the backend program and capture its output
    my_variable = open('my_item.txt').read()
    if my_variable is None:
        my_variable = 'suitcase'
    item_to_detect = ['person', my_variable]
    det = Detector(item_to_detect)
    local_value = open('my_local_footage.txt').read()
    if local_value is None:
        local_value = 'videos/test3.mp4'
    if os.path.exists('doneProcess.txt'):
        os.remove('doneProcess.txt')
    if os.path.exists('targetLocked.txt'):
        os.remove('targetLocked.txt')
    name_list = []
    known_embedding = []
    name_list, known_embedding = det.loadIDFeats()
    print(name_list, known_embedding)
    list_txt(path='name_list.txt', list=name_list)
    fw = open('known_embedding.txt', 'w')
    for line in known_embedding:
        for a in line:
            fw.write(str(a))
            fw.write('\t')
        fw.write('\n')
    fw.close()

    cap = cv2.VideoCapture(local_value)
    fps = int(cap.get(5))
    print('fps:', fps)
    framecounter = 0
    # 初始化帧数计数器
    frame_count = 0
    conf_index = 0
    trackingcounter = 0
    videoWriter = None
    targetLocked = False
    doneProcess = False
    minIndex = None
    trackId = None
    image_paths = []
    list_txt(path='targetLocked.txt', list=targetLocked)
    list_txt(path='doneProcess.txt', list=doneProcess)
    while True:

        success, im = cap.read()
        if im is None:
            break
        # 如果帧数计数器能被5整除，则处理该帧
        if frame_count % 2 == 0:
            DetFeatures = []
            DetFeatures, img_input, box_input = det.loadDetFeats(im)
            result = det.feedCap(im)
# 获取此帧存在的IDs
            current_ids = result['current_ids']
            if len(DetFeatures) > 0 and not targetLocked:
                dist_matrix = _nn_euclidean_distance(known_embedding, DetFeatures, known_embedding[0])
                minimum = np.min(dist_matrix)
                minIndex = dist_matrix.argmin()
                if minimum > 0.12:
                    minIndex = -2
                    # print('最小坐标：', minIndex)
            if (minIndex == conf_index) & (minIndex != -2):
                trackingcounter = trackingcounter + 1
            else:
                conf_index = minIndex
                trackingcounter = 0
            if trackingcounter == 5:
                trackingcounter = 0
                if trackId is None:
                    trackId = current_ids[conf_index]
                    print('trackId:', trackId)
                det.targetTrackId = trackId
                targetLocked = True
                list_txt(path='targetLocked.txt', list=targetLocked)

            result = det.feedCap(im)
            result = result['frame']
            result = imutils.resize(result, height=500)
            if videoWriter is None:
                fourcc = cv2.VideoWriter_fourcc(
                    'm', 'p', '4', 'v')  # opencv3.0
                videoWriter = cv2.VideoWriter(
                    'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

            videoWriter.write(result)
            if det.isLost is True:
                framecounter = framecounter + 1
                print('lost')
            # 每间隔15帧输出一个失主截图
            if framecounter == 10:
                print("输出失主截图")
                cv2.imwrite(f'./test-{det.frameCounter / fps}-second.png', result)
                image_path = f'./test-{det.frameCounter / fps}-second.png'
                image_paths.append(image_path)
                #framecounter = 0

        frame_count = frame_count+1

    # Parse the output to get the video and images
    video_path = 'result.mp4'
    doneProcess = True
    list_txt(path='doneProcess.txt', list=doneProcess)
    # return video_path, image_paths ,不要输出视频了，本地视频无法在dash上播放
    return image_paths


@app.callback(Output('backend-output', 'children'),
              [Input('run-backend-btn', 'n_clicks')])
def update_backend_output(n_clicks):
    if n_clicks:
        #video_path, image_paths = run_backend()
        image_paths = run_backend()
        #video_content = open(video_path, 'rb').read()
        image_contents = [open(path, 'rb').read() for path in image_paths]
        children = [

            *[html.Img(src=f'data:image/png;base64,{base64.b64encode(content).decode()}',
                       style={'width': '25%', 'height': '25%',
                              'display': 'inline-block', 'float': 'right',
                              'position': 'absolute',
                              'top': f'{(i+1)*220}px', 'right': '90px'}) for i, content in enumerate(image_contents)]
        ]
        return children


@app.callback(
    Output('output', 'children'),
    Input('items-dropdown', 'value')
)
def update_output(value):
    my_variable = None
    if value == 'suitcase':
        my_variable = 'suitcase'
    elif value == 'backpack':
        my_variable = 'backpack'
    list_txt(path='my_item.txt', list=my_variable)
    return f'You have selected {value}!'


# Learn more popup
@app.callback(
    Output("markdown", "style"),
    [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")],
)
def update_click_output(button_click, close_click):
    ctx = dash.callback_context
    prop_id = ""
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if prop_id == "learn-more-button":
        return {"display": "block"}
    else:
        return {"display": "none"}

# 定义回调函数，当下拉菜单选项改变时，更新后端变量和输出文本
@app.callback(
    Output('footage-output', 'children'),
    Input('dropdown-footage-selection', 'value')
)
def update_footage_output(selected_value):
    my_selected_value = None
    if selected_value == 'test1':
        local_value = 'videos/test1.mp4'
        my_selected_value = 'https://youtu.be/1hwO2RrHeaA'
    elif selected_value == 'test2':
        local_value = 'videos/test2.mp4'
        my_selected_value = 'https://youtu.be/bpYqqfAGWpk'
    elif selected_value == 'test3':
        local_value = 'videos/test3.mp4'
        my_selected_value = 'https://youtu.be/M1rr89lYkfg'
    list_txt(path='my_footage.txt', list=my_selected_value)
    list_txt(path='my_local_footage.txt', list=local_value)
    return f'You have selected {selected_value}!'

# Footage Selection
@app.callback(
    Output("video-display", "url"),
    [
        Input("dropdown-footage-selection", "value"),
        Input("items-dropdown", "value"),
    ],
)

def load_all_footage(footage, item):
    global url_dict
    url_dict = {  #  实则只有suitcase的视频，后续再添加
        "suitcase": {
            "test1": "https://youtu.be/1hwO2RrHeaA",
            "test2": "https://youtu.be/bpYqqfAGWpk",
            "test3": "https://youtu.be/M1rr89lYkfg",
        },
        "backpack": {
            "test1": "https://youtu.be/RM9j3RKexSg",
            "test2": "https://youtu.be/EeNT57Y9_hE",
            "test3": "https://youtu.be/0nqDW_tMn58",
        },
    }
    url = url_dict[item][footage]
    return url

if __name__ == '__main__':
    app.run_server(debug=True)
