import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import base64

app = dash.Dash()

url = ['https://youtu.be/1hwO2RrHeaA',
       'https://youtu.be/bpYqqfAGWpk',
       'https://youtu.be/M1rr89lYkfg',
       'https://youtu.be/RM9j3RKexSg',
       'https://youtu.be/RM9j3RKexSg',
       'https://youtu.be/EeNT57Y9_hE',
       'https://youtu.be/EeNT57Y9_hE',
       'https://youtu.be/0nqDW_tMn58',
       'https://youtu.be/0nqDW_tMn58']
# 定义八个视频播放器
videos = []
for i in range(8):
    video = html.Video(
        controls=True,
        src=url[i],
        style={'width': '50%', 'height': 'auto', 'margin-top': '10px', 'margin-right': '3px'}
    )
    videos.append(video)

# 定义上传照片的框
photos = []
for i in range(6):
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
            'width': '60%',
            'height': '150px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '50px',
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
        labelStyle={'display': 'inline-block', 'margin-right': '3px'}
    )
    video_selectors.append(selector)

# 定义页面布局
app.layout = html.Div(children=[
    html.Div([
        html.Div([
            html.Div([videos[0],
                video_selectors[0],
                videos[1],
                video_selectors[1]
            ], className='three columns'),
            html.Div([videos[2],
                video_selectors[2],
                videos[3],
                video_selectors[3]
            ], className='three columns'),
            html.Div([videos[4],
                      video_selectors[4],
                      videos[5],
                      video_selectors[5]
            ], className='three columns'),
            html.Div([videos[6],
                      video_selectors[6],
                      videos[7],
                      video_selectors[7]
            ], className='three columns'),
        ], className='row')
    ], className='eleven columns'),
    html.Div([
        html.Div([photos[0],
            photos[1],
            photos[2]
        ], className='four columns'),
        html.Div([photos[3],
            photos[4],
            photos[5]
        ], className='four columns')
    ], className='five columns'),
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
@app.callback(Output('upload-photo-1', 'children'),
              [Input('upload-photo-1', 'contents')],
              [State('upload-photo-1', 'filename')])
def display_uploaded_photo(contents, filename):
    if contents is not None:
        # 将上传的照片转换为base64编码格式
        encoded_image = base64.b64encode(contents.encode('utf-8'))
        return html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                     style={'width': '100%', 'height': 'auto'})
        ])
    else:
        return html.Div([
            html.Div([
                html.I(className='fas fa-plus'),
                html.Span('添加照片', className='add-photo-label')
            ], className='upload-button')
        ], className='upload-area')

# 运行app
if __name__ == '__main__':
    app.run_server(debug=True)
