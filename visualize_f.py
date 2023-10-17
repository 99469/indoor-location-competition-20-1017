import plotly.graph_objs as go
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import xlwt
import pandas as pd


def save_figure_to_html(fig, filename):
    fig.write_html(filename)


def visualize_trajectory(trajectory, floor_plan_filename, width_meter, height_meter, title=None,
                         mode='lines + markers + text', show=False):
    fig = go.Figure()

    # add trajectory
    size_list = [6] * trajectory.shape[0]
    size_list[0] = 10
    size_list[-1] = 10

    color_list = ['rgba(4, 174, 4, 0.5)'] * trajectory.shape[0]
    color_list[0] = 'rgba(12, 5, 235, 1)'
    color_list[-1] = 'rgba(235, 5, 5, 1)'

    position_count = {}
    text_list = []
    for i in range(trajectory.shape[0]):
        if str(trajectory[i]) in position_count:
            position_count[str(trajectory[i])] += 1
        else:
            position_count[str(trajectory[i])] = 0
        text_list.append('        ' * position_count[str(trajectory[i])] + f'{i}')
    text_list[0] = 'Start Point: 0'
    text_list[-1] = f'End Point: {trajectory.shape[0] - 1}'

    fig.add_trace(
        go.Scattergl(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            mode=mode,
            marker=dict(size=size_list, color=color_list),
            line=dict(shape='linear', color='rgb(100, 10, 100)', width=2, dash='dot'),
            text=text_list,
            textposition="top center",
            name='trajectory',
        ))

    # add floor plan
    floor_plan = Image.open(floor_plan_filename)
    fig.update_layout(images=[
        go.layout.Image(
            source=floor_plan,
            xref="x",
            yref="y",
            x=0,
            y=height_meter,
            sizex=width_meter,
            sizey=height_meter,
            sizing="contain",
            opacity=1,
            layer="below",
        )
    ])

    # configure
    fig.update_xaxes(autorange=False, range=[0, width_meter])
    fig.update_yaxes(autorange=False, range=[0, height_meter], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=go.layout.Title(
            text=title or "No title.",
            xref="paper",
            x=0,
        ),
        autosize=True,
        width=900,
        height=200 + 900 * height_meter / width_meter,
        template="plotly_white",
    )

    if show:
        fig.show()

    return fig


def k_means(data, value):
    # 读取原始数据
    X = data
    # 转化为numpy array
    print(X)
    X = np.array(X)
    print(X)

    # 类簇的数量
    n_clusters = 5

    # 开始调用函数聚类
    cls = KMeans(n_clusters).fit(X)

    # 输出X中每项所属分类的一个列表
    print(cls.labels_)

    # 画图
    markers = ['*', 'o', '+', 's', 'v']

    for i in range(n_clusters):
        members = cls.labels_ == i  # members是布尔数组
        # array([False, False, False, False, False, False, False, False, False,
        #    False, False, False, False, False, False, False, False, False,
        #    False, False,  True, False, False, False, False, False, False,
        #    False,  True,  True,  True,  True, False, False,  True,  True,
        #     True, False,  True, False,  True,  True,  True,  True,  True,
        #     True, False, False, False, False,  True,  True,  True, False,
        #    False, False, False,  True,  True,  True,  True,  True,  True,
        #    ...])
        plt.scatter(X[members, 0], X[members, 1], s=60, marker=markers[i], c='b', alpha=0.5)  # 画与menbers数组中匹配的点

    plt.title('China')
    plt.show()


def save_excel(position, value):
    wb = xlwt.Workbook()
    # 添加一个表
    ws = wb.add_sheet('data')
    for i in range(len(position)):
        for j in range(4):
            if j == 0:
                ws.write(i, j, i)
            elif j != 3:
                ws.write(i, j, position[i][j - 1])
            else:
                ws.write(i, j, value[i])

    # 保存excel文件
    wb.save('./test.xls')


def visualize_heatmap(position, value, floor_plan_filename, width_meter, height_meter, colorbar_title="colorbar",
                      title=None, show=False):
    fig = go.Figure()
    color_list = ['red']
    is_final = title == 'Magnetic Strength'
    if is_final:

        save_excel(position, value)
        color = ['red', 'orange', 'blue']
        df_raw1 = pd.read_excel('./kmeans/datalabels.xlsx', sheet_name=0, index_col=0)
        column_data = df_raw1[0].tolist()
        for item in column_data:
            color_list.append(color[item])
    # add heat map
    fig.add_trace(
        go.Scatter(x=position[:, 0],
                   y=position[:, 1],
                   mode='markers',
                   marker=dict(size=7,
                               color=color_list if is_final else value,
                               colorbar=dict(title=colorbar_title),
                               colorscale="Rainbow"),
                   text=value,
                   name=title))

    # add floor plan
    floor_plan = Image.open(floor_plan_filename)
    fig.update_layout(images=[
        go.layout.Image(
            source=floor_plan,
            xref="x",
            yref="y",
            x=0,
            y=height_meter,
            sizex=width_meter,
            sizey=height_meter,
            sizing="contain",
            opacity=1,
            layer="below",
        )
    ])

    # configure
    fig.update_xaxes(autorange=False, range=[0, width_meter])
    fig.update_yaxes(autorange=False, range=[0, height_meter], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=go.layout.Title(
            text=title or "No title.",
            xref="paper",
            x=0,
        ),
        autosize=True,
        width=900,
        height=200 + 900 * height_meter / width_meter,
        template="plotly_white",
    )
    if show:
        fig.show()

    return fig


def visualize_heatmap_with_centers(position, value, centers, floor_plan_filename, width_meter, height_meter,
                      colorbar_title="colorbar",
                      title=None, show=False):
    fig = go.Figure()
    color_list = ['red']
    is_final = title == 'Magnetic Strength'
    if is_final:

        save_excel(position, value)
        color = ['red', 'orange', 'blue']
        df_raw1 = pd.read_excel('./kmeans/datalabels.xlsx', sheet_name=0, index_col=0)
        column_data = df_raw1[0].tolist()
        for item in column_data:
            color_list.append(color[item])
    # add heat map
    fig.add_trace(
        go.Scatter(x=position[:, 0],
                   y=position[:, 1],
                   mode='markers',
                   marker=dict(size=7,
                               color=color_list if is_final else value,
                               colorbar=dict(title=colorbar_title),
                               colorscale="Rainbow"),
                   text=value,
                   name=title))

    # add centers
    fig.add_trace(
        go.Scattergl(
            x=centers[:, 0],
            y=centers[:, 1],
            mode='text',
            marker=dict(size=12, color='rgba(255, 0, 0, 1)'),
            name='centers',
            text='★'  # Unicode for five-pointed star
            # textposition='top center'  # 调整文本位置为中上方
        ))

    # add floor plan
    floor_plan = Image.open(floor_plan_filename)
    fig.update_layout(images=[
        go.layout.Image(
            source=floor_plan,
            xref="x",
            yref="y",
            x=0,
            y=height_meter,
            sizex=width_meter,
            sizey=height_meter,
            sizing="contain",
            opacity=1,
            layer="below",
        )
    ])

    # configure
    fig.update_xaxes(autorange=False, range=[0, width_meter])
    fig.update_yaxes(autorange=False, range=[0, height_meter], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=go.layout.Title(
            text=title or "No title.",
            xref="paper",
            x=0,
        ),
        autosize=True,
        width=900,
        height=200 + 900 * height_meter / width_meter,
        template="plotly_white",
    )

    if show:
        fig.show()

    return fig
