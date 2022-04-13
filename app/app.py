from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import uuid

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def hello_world():
	request_type = request.method
	if request_type == 'GET':
		return render_template('index.html', href='static/base.svg')
	else:
		text = request.form['text']
		random_name = uuid.uuid4().hex
		path='app/static/'+random_name+'.svg'
		model = load('app/model.joblib')
		data_input = create_input(text)
		make_picture('app/AgesAndHeights.pkl', model,data_input,path)
		return render_template('index.html', href=path[4:])

def make_picture(data, model, new_input, output_file):
	data = pd.read_pickle(data)
	data = data[data['Age']>0]
	ages = data['Age']
	heights = data['Height']

	x_new = np.array(list(range(19))).reshape(19,1)
	preds = model.predict(x_new)

	fig = px.scatter(x=ages, y=heights, title="Heights vs Age of people", labels={'x':'Age (years)', 'y':'Heights (inches)'})
	fig.add_trace(go.Scatter(x=x_new.reshape(19), y=preds, mode='lines', name='Model'))

	new_preds = model.predict(new_input)
	fig.add_trace(go.Scatter(x=new_input.reshape(len(new_input)), y=new_preds, name='New outputs', mode='markers', marker={'color':'purple','size':20}))
	fig.write_image(output_file, width=800, engine='kaleido')

def create_input(floats_str):
	def is_float(x):
		try:
			float(x)
			return(True)
		except:
			return(False)
	floats = [float(x) for x in floats_str.split(',') if is_float(x)]
	return np.array(floats).reshape(len(floats), 1)



