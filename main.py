import streamlit as st
import numpy as np 
import seaborn as sns
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 
import plotly.express as px
#From sklearn import datasets
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import LabelEncoder
from PIL import Image

matplotlib.use('agg')#to write into the file rather than displaying in the window

st.set_option('deprecation.showPyplotGlobalUse', False)#to avoid warnings

st.title("AI Cumulate")

st.write('''## An interactive data app with streamlit''')

def main():
	activities=['About Us','EDA','Visualisation','Model','Covid Tracker']
	option=st.sidebar.selectbox('Select',activities)

	#Dealing with EDA
	if option=='EDA':
		if st.checkbox('Use Datasets from sklearn'):
			st.write('Select your dataset')
			dataset_name=""
			dataset_name = st.selectbox('Select Dataset', ('Breast Cancer','Wine','Iris'))

			data=None
			def get_dataset(name):
				if name=='Iris':
					data=datasets.load_iris()
				elif name=='Wine':
					data=datasets.load_wine()
				else:
					data=datasets.load_breast_cancer()
				x=data.data
				y=data.target
				z=data.feature_names
				return x,y,z

			x,y,z = get_dataset(dataset_name)
			df=pd.DataFrame(x,columns=z)
			df1=df
			st.dataframe(df.head(50))

			if st.checkbox("Display Shape"):
				st.write(df.shape)

			if st.checkbox("Display Columns"):
				st.write(df.columns)

			if st.checkbox("Select Multiple Columns"):
				selected_columns = st.multiselect("Select the columns",df.columns)
				df1=df[selected_columns]
				st.dataframe(df1.head(50))

			if st.checkbox("Display Summary"):
				if selected_columns is not None:
					st.write(df1.describe().T)
				else:
					st.write(df.describe().T)

			if st.checkbox("Display Null Values"):
				st.write(df.isnull().sum())

			if st.checkbox("Display the various data types"):
				st.write(df.dtypes)

			if st.checkbox("Display correlation with respect to various columns"):
				st.write(df.corr())


		if st.checkbox('Use Your Datasets'):
			st.write('''### Let's explore different Classifiers and Dataset''')
			st.subheader('Exploratory Data Analysis')
			data=st.file_uploader("Upload Dataset",type=['csv','xlsx','txt','json'])
			selected_columns=[]

			if data is not None:
				st.success("Successfully Loaded!")
				df=pd.read_csv(data)
				df1=df
				st.dataframe(df.head(50))

				if st.checkbox("Display Shape"):
					st.write(df.shape)

				if st.checkbox("Display Columns"):
					st.write(df.columns)

				if st.checkbox("Select Multiple Columns"):
					selected_columns = st.multiselect("Select the columns",df.columns)
					df1=df[selected_columns]
					st.dataframe(df1.head(50))

				if st.checkbox("Display Summary"):
					if selected_columns is not None:
						st.write(df1.describe().T)
					else:
						st.write(df.describe().T)

				if st.checkbox("Display Null Values"):
					st.write(df.isnull().sum())

				if st.checkbox("Display the various data types"):
					st.write(df.dtypes)

				if 	st.checkbox("Display correlation with respect to various columns"):
					st.write(df.corr())

	#Dealing with Visualization
	elif option=="Visualisation":
		if st.checkbox('Use Datasets From Sklearn'):
			st.write('Select your dataset')
			dataset_name=""
			dataset_name = st.selectbox('Select Dataset', ('Breast Cancer','Wine','Iris'))

			data=None
			def get_dataset(name):
				if name=='Iris':
					data=datasets.load_iris()
				elif name=='Wine':
					data=datasets.load_wine()
				else:
					data=datasets.load_breast_cancer()
				x=data.data
				y=data.target
				z=data.feature_names
				return x,y,z

			x,y,z = get_dataset(dataset_name)
			df=pd.DataFrame(x,columns=z)

			if(dataset_name):
				st.dataframe(df)
				st.write("The shape of your dataset is:",x.shape)
				st.write("Unique target variables are",len(np.unique(y)))

			selected_columns=[]

			if st.checkbox('Select multiple columns to plot'):
				selected_columns = st.multiselect("Select the columns",df.columns)
				df1=df[selected_columns]
				st.dataframe(df1.head(50))

			if st.checkbox('Heatmap'):
				if selected_columns is not None:
					st.write(sns.heatmap(df1.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
				else:
					st.write(sns.heatmap(df.corr(),vmax=1,square=True,annot=True,cmap='viridis'))				
				st.pyplot()

			if st.checkbox('Pair Plot'):
				if selected_columns is not None:
					st.write(sns.pairplot(df1,diag_kind='kde'))
				else:
					st.write(sns.pairplot(df,diag_kind='kde'))
				st.pyplot()

			if st.checkbox('Display Pie Chart'):
				all_columns = df.columns.to_list()
				pie_columns = st.selectbox("select column to display",all_columns)
				pieChart = df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
				st.write(pieChart)
				st.pyplot()


		if st.checkbox('Use Your Datasets'):
			st.write('''### Let's explore different Classifiers and Dataset''')
			st.subheader("Data Visualisation")
			data=st.file_uploader("Upload Dataset",type=['csv','xlsx','txt','json'])
			selected_columns=[]

			if data is not None:
				st.success("Successfully Loaded!")
				df=pd.read_csv(data)
				df1=df
				st.dataframe(df.head(50))

				if st.checkbox('Select multiple columns to plot'):
					selected_columns = st.multiselect("Select the columns",df.columns)
					df1=df[selected_columns]
					st.dataframe(df1.head(50))

				if st.checkbox('Heatmap'):
					if selected_columns is not None:
						st.write(sns.heatmap(df1.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
					else:
						st.write(sns.heatmap(df.corr(),vmax=1,square=True,annot=True,cmap='viridis'))				
					st.pyplot()

				if st.checkbox('Pair Plot'):
					if selected_columns is not None:
						st.write(sns.pairplot(df1,diag_kind='kde'))
					else:
						st.write(sns.pairplot(df,diag_kind='kde'))
					st.pyplot()

				if st.checkbox('Display Pie Chart'):
					all_columns = df.columns.to_list()
					pie_columns = st.selectbox("select column to display",all_columns)
					pieChart = df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
					st.write(pieChart)
					st.pyplot()

	#Dealing with Model Building
	elif option=="Model":
		st.write('''### Let's explore different Classifiers and Dataset''')
		st.subheader("Model Building")
		if st.checkbox('Use Datasets From Sklearn'):
			st.write('Select your dataset')
			dataset_name=""
			dataset_name = st.selectbox('Select Dataset', ('Breast Cancer','Wine','Iris'))

			data=None
			def get_dataset(name):
				if name=='Iris':
					data=datasets.load_iris()
				elif name=='Wine':
					data=datasets.load_wine()
				else:
					data=datasets.load_breast_cancer()
				x=data.data
				y=data.target
				z=data.feature_names
				return x,y,z

			x,y,z = get_dataset(dataset_name)
			df=pd.DataFrame(x,columns=z)

			# Generating the Dataset
			if(dataset_name):
				st.dataframe(df)
				st.write("The shape of your dataset is:",x.shape)
				st.write("Unique target variables are",len(np.unique(y)))

			if st.checkbox('Select Multiple Columns'):
				selected_columns = st.multiselect('Select the columns',df.columns)
				df1=df[selected_columns]
				st.dataframe(df1.head(50))
				flag=1
		

			seed = st.sidebar.slider('Seed',1,200)
			classifier_name = st.sidebar.selectbox('Select your preferred classifier',('KNN','SVM','Logistic Regression','Naive Bayes','Random Forest'))

			def add_parameter(name_of_clf):
				params=dict()
				if name_of_clf=="SVM":
					C=st.sidebar.slider('C',0.01,15.0)
					params['C']=C
				elif name_of_clf=="KNN":
					K=st.sidebar.slider('K',1,25)
					params['K']=K
				return params

			params=add_parameter(classifier_name)

			#After defining the paramters, we now create the model
			def get_classifier(name_of_clf,params):
				clf=None
				if name_of_clf=='SVM':
					clf=SVC(C=params['C'])
				elif name_of_clf=='KNN':
					clf=KNeighborsClassifier(n_neighbors=params['K'])
				elif name_of_clf=='Logistic Regression':
					clf=LogisticRegression()
				elif name_of_clf=='Naive Bayes':
					clf=GaussianNB()
				elif name_of_clf=="Random Forest":
					clf=RandomForestClassifier()
				else:
					st.warning('Select your choice of algorithm')
				return clf 

			clf=get_classifier(classifier_name,params)	

			x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=seed)
			clf.fit(x_train,y_train)
			y_pred=clf.predict(x_test)
			accuracy = accuracy_score(y_test,y_pred)
			st.write('Name of classifier is',classifier_name)
			st.write(f'Accuracy for your model is {accuracy*100:.2f}%')		
	

		elif st.checkbox('Use my own dataset'):
			data = st.file_uploader("Upload Dataset",type=['csv','xlsx','txt','json'])
			selected_columns=[]

			if data is not None:
				st.success("Data Successfully Loaded")
				df=pd.read_csv(data)
				df1=df
				st.dataframe(df.head(50))

				if st.checkbox('Select Multiple Columns'):
					selected_columns = st.multiselect('Select the columns',df.columns)
					df1=df[selected_columns]
					st.dataframe(df1.head(50))
					flag=1

				#Dividing my data into X and Y variables
				x=df.iloc[:,0:-1]
				y=df.iloc[:,-1]

				seed = st.sidebar.slider('Seed',1,200)
				classifier_name = st.sidebar.selectbox('Select your preferred classifier',('KNN','SVM','Logistic Regression','Naive Bayes','Random Forest'))
				
				#Now we define a function add parameters for the respective models
				def add_parameter(name_of_clf):
					params=dict()
					if name_of_clf=="SVM":
						C=st.sidebar.slider('C',0.01,15.0)
						params['C']=C
					elif name_of_clf=="KNN":
						K=st.sidebar.slider('K',1,25)
						params['K']=K
					return params

				params=add_parameter(classifier_name)			

				#After defining the paramters, we now create the model
				def get_classifier(name_of_clf,params):
					clf=None
					if name_of_clf=='SVM':
						clf=SVC(C=params['C'])
					elif name_of_clf=='KNN':
						clf=KNeighborsClassifier(n_neighbors=params['K'])
					elif name_of_clf=='Logistic Regression':
						clf=LogisticRegression()
					elif name_of_clf=='Naive Bayes':
						clf=GaussianNB()
					elif name_of_clf=="Random Forest":
						clf=RandomForestClassifier()
					else:
						st.warning('Select your choice of algorithm')
					return clf 

				clf=get_classifier(classifier_name,params)

				x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=seed)
				clf.fit(x_train,y_train)
				y_pred=clf.predict(x_test)
				accuracy = accuracy_score(y_test,y_pred)
				st.write('Name of classifier is',classifier_name)
				st.write(f'Accuracy for your model is {accuracy*100:.2f}%')

	#About page
	elif option=="Covid Tracker":

		st.write("# Covid Tracker")
		st.write('An animation created using plotly library which shows the rise of Covid 19 as a function of time')
		covid = pd.read_csv("https://raw.githubusercontent.com/shinokada/covid-19-stats/master/data/daily-new-confirmed-cases-of-covid-19-tests-per-case.csv");
		covid.columns = ['Country','Code','Date','Confirmed','Days Since Confirmed']
		#Here we basically convert date time from string in the original dataset to a value which we can use in the plot.
		covid['Date'] = pd.to_datetime(covid['Date']).dt.strftime('%Y-%m-%d')

		st.write(covid)

		country_options = covid['Country'].unique().tolist()
		date_options = covid['Date'].unique().tolist()
		date=st.selectbox("Which date would you like to see?",date_options,100)
		country=st.multiselect("Which country would you like to see?",country_options,['India'])

		covid=covid[covid['Country'].isin(country)]
		#covid=covid[covid['Date']==date]

		fig2 = px.bar(covid,x="Country",y="Confirmed",color="Country",range_y=[0,35000],
			animation_frame="Date",animation_group="Country")

		fig2.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 30
		fig2.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5

		fig2.update_layout(width=800)

		st.write(fig2)

	else:
		st.markdown(' An interactive web app that has been created by infusing my knowledge of fundamental Machine Learning and Web Development Concepts ')
		st.markdown(' AI Cumulate greatly simplifies the process of writing ML Code')
		st.markdown(' For example if a user wants to make analysis or predictions on a dataset, all he has to do is upload the dataset, select his preferred algorithm of choice along with the respective parameters and the code at the backend would give the user the appropriate result along with the accuracy of the model.')
		st.markdown(' The applications of this app can range from clients showcasing their model and observations to investors or just helping our friends understand the usage of Machine Learning models without displaying the laborious code at the backend.')
		st.markdown(' In addition to this with the current pandemic going on, it was of relevance to do something related to covid, henceforth a CoVid tracker has been incorporated which accurately describes the no. of cases in different parts of the world as a function of time.')
		st.balloons()


if __name__ == '__main__':
	main()