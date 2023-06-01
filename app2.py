import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import base64
import dash_player as player
import os

app = dash.Dash()

# 设置上传文件夹
UPLOAD_DIRECTORY = "./images/origin"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

url = ['https://youtu.be/1hwO2RrHeaA',
       'https://youtu.be/bpYqqfAGWpk',
       'https://youtu.be/M1rr89lYkfg',
       'https://youtu.be/RM9j3RKexSg',
       'https://youtu.be/RM9j3RKexSg',
       'https://youtu.be/EeNT57Y9_hE',
       'https://youtu.be/EeNT57Y9_hE',
       'https://youtu.be/0nqDW_tMn58']

# 定义八个视频播放器
videos = []
for i in range(8):
    video = player.DashPlayer(
        id="video-display{}".format(i+1),
        controls=True,
        playing=False,
        volume=1,
        url=url[i],
        width="150px",
        height="120px",
    )
    videos.append(video)

# 定义上传照片的框
photos = []
for i in range(4):
    photo = dcc.Upload(
        id='upload-photo-{}'.format(i+1),
        children=html.Div([
            html.Div([
                html.Span('+'),
            ], className='upload-message'),
            html.Div([
                html.I(className='fas fa-plus'),
                html.Span('添加照片', className='add-photo-label')
            ], className='upload-button')
        ]),
        style={
            'width': '50%',
            'height': '123px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'position': 'relative',
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center'
        },
        multiple=False
    )
    photos.append(photo)

# 定义视频选择器的布局和回调函数
video_selectors = []
for i in range(8):
    selector = dcc.Checklist(
        id='video-selector-{}'.format(i+1),
        options=[{'label': 'Video {}'.format(i+1), 'value': i+1}],
        value=[],
        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
    )
    video_selectors.append(selector)

# 定义页面布局
app.layout = html.Div([
    html.Div([
        html.Button('Run', id='run-button', n_clicks=0),
        html.Button('Delete', id='delete-button', n_clicks=0)
    ], style={'text-align': 'right', 'margin-top': '10px', 'margin-bottom': '1px'}),

    html.Div(id='video-selection',
             style={'display': 'inline-block', 'margin-top': '10px',
                    'margin-bottom': '1px',
                    'text-align': 'right'}),
    html.Div([
        html.Div([
            html.Div([videos[0],
                video_selectors[0],
                videos[1],
                video_selectors[1]
            ], className='two columns',style={'margin-left': '50px', 'margin-top': '50px'}),
            html.Div([videos[2],
                      video_selectors[2],
                      videos[3],
                      video_selectors[3]
            ], className='two columns', style={'margin-left': '20px', 'margin-top': '50px'}),
            html.Div([videos[4],
                      video_selectors[4],
                      videos[5],
                      video_selectors[5]
            ], className='two columns', style={'margin-left': '20px', 'margin-top': '50px'}),
            html.Div([videos[6],
                      video_selectors[6],
                      videos[7],
                      video_selectors[7]
                      ], className='two columns', style={'margin-left': '20px', 'margin-top': '50px'}),
            html.Div([photos[0],
                      photos[1],
                      ], className='two columns', style={'margin-left': '55px', 'margin-top': '50px'}),
            html.Div([photos[2],
                      photos[3],
                      ], className='two columns', style={'margin-left': '-15px', 'margin-top': '50px'})
        ], className='row'),
    ], className='fifteen columns'),
    html.Div(id='video-selection',
             style={'display': 'inline-block', 'margin-top': '10px',
                    'margin-bottom': '1px',
                    'text-align': 'right'}),
])

# 处理视频选择器的回调函数
@app.callback(Output('video-selection', 'children'),
              [Input('video-selector-1', 'value'),
               Input('video-selector-2', 'value'),
               Input('video-selector-3', 'value'),
               Input('video-selector-4', 'value'),
               Input('video-selector-5', 'value'),
               Input('video-selector-6', 'value'),
               Input('video-selector-7', 'value'),
               Input('video-selector-8', 'value')])
def handle_video_selection(v1, v2, v3, v4, v5, v6, v7, v8):
    selected_videos = []
    if v1:
        selected_videos.append(videos[0])
    if v2:
        selected_videos.append(videos[1])
    if v3:
        selected_videos.append(videos[2])
    if v4:
        selected_videos.append(videos[3])
    if v5:
        selected_videos.append(videos[4])
    if v6:
        selected_videos.append(videos[5])
    if v7:
        selected_videos.append(videos[6])
    if v8:
        selected_videos.append(videos[7])
    # 调用后端处理程序处理选中的视频
    # process_selected_videos(selected_videos)
    return selected_videos

# 处理上传照片的回调函数
@app.callback(
    [Output('upload-photo-1', 'children'),
     Output('upload-photo-2', 'children'),
     Output('upload-photo-3', 'children'),
     Output('upload-photo-4', 'children')],
    [Input('upload-photo-1', 'contents'),
     Input('upload-photo-2', 'contents'),
     Input('upload-photo-3', 'contents'),
     Input('upload-photo-4', 'contents')],
    [State('upload-photo-1', 'filename'),
     State('upload-photo-2', 'filename'),
     State('upload-photo-3', 'filename'),
     State('upload-photo-4', 'filename')])
def update_uploaded_images(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = ''
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    images = []
    for i, (content, filename) in enumerate(zip(args[:4], args[4:])):
        if button_id == f'upload-photo-{i+1}' and content is not None:
            # Decode the base64-encoded photo data
            photo_data = base64.b64decode(content.split(',')[1])
            # Save the photo to the "origin" folder
            with open(os.path.join(UPLOAD_DIRECTORY, filename), 'wb') as f:
                f.write(photo_data)
            # Create an HTML image tag to display the photo
            img_tag = html.Img(src=content, style={'max-width': '100%', 'max-height': '100%'})
            images.append(img_tag)
        else:
            if content is not None:
                # If the button was not clicked or no new image was uploaded, show the existing image
                img_tag = html.Img(src=content, style={'max-width': '100%', 'max-height': '100%'})
                images.append(img_tag)
            else:
                # If no image was uploaded, show the upload button
                images.append(dcc.Upload('Upload Photos', style={'width': '100%', 'height': '100%'}))
    return images
# 运行app
if __name__ == '__main__':
    app.run_server(debug=True)
