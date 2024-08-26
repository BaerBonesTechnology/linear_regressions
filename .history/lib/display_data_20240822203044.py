#@title Train Model using linear regression

# general 
import io

# data
import pandas as pd


# data visualization
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# constants
import lib.constants.labels as labels
import lib.constants.graph_type  as gt

def make_plots(df, feature_names, label_name, model_output, sample_size=200):

    # get random sample of data
    random_sample = df.sample(n=sample_size).copy()
    random_sample.reset_index()
    weights, bias, epochs, rmse = model_output

    # create figure
    is_2d_plot = len(feature_names) == 1
    model_plot_type = gt.scatter if is_2d_plot else gt.SURFACE_GRAPH
    fig = make_subplots(rows = 1, cols = 2, subplot_titles=("Loss Curve", "Model Plot"),
                        specs=[[{"type": gt.scatter}, {"type": model_plot_type}]])
    plot_data(random_sample, feature_names, label_name, fig)
    plot_model(random_sample, feature_names, weights, bias, fig)
    plot_loss_curve(epochs, rmse, fig)

    fig.show()
    return

def plot_loss_curve(epochs, rmse, fig):
    curve = px.line(x=epochs, y=rmse)
    curve.update_traces(line_color='#636EFA', line_width=3)

    fig.append_trace(curve.data[0], row=1, col=1)
    fig.update_xaxes(title_text= labels.epoch_label, row=1, col=1)
    fig.update_yaxes(title_text= labels.rmse_label, row=1, col=1)

    return

def plot_data(df, feature_names, label_name, fig):
    if len(feature_names) ==1:
        scatter = px.scatter(df, x = feature_names[0], y=label_name)
    else:
        scatter = px.scatter_3d(df, x=feature_names[0], y=feature_names[1], z=label_name)

    fig.append_trace(scatter.data[0], row=1, col=2)
    if(len(feature_names) == 1):
        fig.update_xaxes(title_text=feature_names[0], row=1, col=2)
        fig.update_yaxes(title_text=label_name, row=1, col=2)
    else:
        fig.update_layout(scene1 = dict(xaxis_title=feature_names[0], yaxis_title=feature_names[1], zaxis_title=label_name))

    return

def plot_model(df, features, weights, bias, fig):
  df['FARE_PREDICTED'] = bias[0]

  for index, feature in enumerate(features):
    df['FARE_PREDICTED'] = df['FARE_PREDICTED'] + weights[index][0] * df[feature]

  if len(features) == 1:
    model = px.line(df, x=features[0], y='FARE_PREDICTED')
    model.update_traces(line_color='#ff0000', line_width=3)
  else:
    z_name, y_name = "FARE_PREDICTED", features[1]
    z = [df[z_name].min(), (df[z_name].max() - df[z_name].min()) / 2, df[z_name].max()]
    y = [df[y_name].min(), (df[y_name].max() - df[y_name].min()) / 2, df[y_name].max()]
    x = []
    for i in range(len(y)):
      x.append((z[i] - weights[1][0] * y[i] - bias[0]) / weights[0][0])

    plane=pd.DataFrame({'x':x, 'y':y, 'z':[z] * 3})

    light_yellow = [[0, '#89CFF0'], [1, '#FFDB58']]
    model = go.Figure(data=go.Surface(x=plane['x'], y=plane['y'], z=plane['z'],
                                      colorscale=light_yellow))

  fig.add_trace(model.data[0], row=1, col=2)

  return

def model_info(feature_names, label_name, model_output):
  weights = model_output[0]
  bias = model_output[1]

  nl = "\n"
  header = "-" * 80
  banner = header + nl + "|" + "MODEL INFO".center(78) + "|" + nl + header

  info = ""
  equation = label_name + " = "

  for index, feature in enumerate(feature_names):
    info = info + "Weight for feature[{}]: {:.3f}\n".format(feature, weights[index][0])
    equation = equation + "{:.3f} * {} + ".format(weights[index][0], feature)

  info = info + "Bias: {:.3f}\n".format(bias[0])
  equation = equation + "{:.3f}\n".format(bias[0])

  return banner + nl + info + nl + equation

print("SUCCESS: defining plotting functions complete.")