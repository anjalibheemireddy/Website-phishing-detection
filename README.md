# Website-phishing-detection-using Machine Learning
#Note-The code is executed in Jupyter Notebook Environment and the data set File is loaded by inserting the file path wherever necessary.

Online security is crucial in today's linked digital world. Phishing attacks, which attempt to trick users into disclosing personal information on phoney websites, present a serious concern.
To prevent people from falling for phishing scams, our project, "Website Phishing Detection with Machine Learning," uses machine learning approaches to address this issue.
Our main objective is to offer a simple and useful solution that enables people and companies to recognise potentially harmful websites quickly. 
The project's accessibility and simplicity are its core components. Users do not need to be specialists in cybersecurity to use our system. 
Our approach identifies probable phishing websites by inspecting various website properties, including URL structure, port, SSL certificates, and hyperlinks.
Our program allows users to rapidly determine the integrity of any strange URLs they come across. 

PROJECT DESIGN:
Hardware and Software Requirements:

•	     Processor: Any Update Processer

•	     RAM: Min 4 GB

•	    Hard Disk: Min 100 GB

•	     Operating System: Windows family

•	     Technology: Python 3.6

•	     IDE: Jupyter Notebook

Technologies and Design Methods:

•	Python is one of the programming languages (with libraries like Scikit-learn, Pandas, and NumPy)

•	Processing and Analyzing Data: Jupyter Notebooks, Matplotlib, Seaborn 

•	Machine Learning Models: Decision Trees, Naive Bayes, K-Nearest Neighbors (KNN), Neural Networks

After trying out different regression and classification models we found that the following have the most accuracy among all.
-K-Nearest Neighbours (KNN) algorithm
-Decision Trees
-the Naive Bayes classifier 
-Neural Networks 

For prediction analysis:

Feature extraction: A method that accepts a URL as input was developed. From the provided URL, this method retrieves a number of characteristics. These characteristics—such as the URL's length, the existence of characters, and more—act as indicators or clues regarding its nature.

Machine Learning Model: Pickle was used to load a pre-trained machine learning model. A sizable dataset of URLs has already been used to train this model. It developed the ability to identify patterns in URLs that are frequently connected to phishing attempts throughout its training.

Feature Dictionary: The features we extract from the user-provided URL are stored in a dictionary we have put up named features1. This dictionary gives each feature a numerical number that expresses its attributes.
 
Comparison: We compared the feature names in features1 with the feature names the model was trained on to make sure our extracted features meet the expectations of the model. For input compatibility, this step is essential.

Prediction: We provided the extracted features as input data for the pre-trained machine-learning model as soon as we were certain that our features matched the model's expectations. The format of this data was the same as that of the data used to train the model.

We have found that the decision tree model has more accuracy in giving weather the Url is legitimate or not.
In future we can try out more models and try out different data features to get more accuracy of the model. We can also add a small UI to show case how it can work or build a website that can work in real time and deploy it.


REFERENCES:

Scikit-Learn Documentation and Tutorials: Scikit-Learn is an essential machine-learning library for Python. We will refer to the official documentation for model development and utilize tutorials like the one on Kaggle's website for hands-on guidance.
https://scikit-learn.org/stable/documentation.html
Kaggle reference to introduction to machine learning and building models.
https://www.kaggle.com/learn/intro-to-machine-learning
Dataset sources and relevant documentation
https://data.mendeley.com/datasets/c2gw7fy2j4/3- The main data source we have used in our project.
https://www.sciencedirect.com/science/article/abs/pii/S0952197621001950?via%3Dihub
Technical Resources:
Phishing Web Page Detection Methods: URL and HTML Features Detection | IEEE Conference Publication | IEEE Xplore-a research paper on how to select features of the URL and importance
Creating Phishing page of a website - GeeksforGeeks-How to build a website using html.
An effective detection approach for phishing websites using URL and HTML features | Scientific Reports (nature.com)-reference to paper on how to classify the features.
Data set citation:
Hannousse, Abdelhakim; Yahiouche, Salima (2021), “Web page phishing detection”, Mendeley Data, V3, doi: 10.17632/c2gw7fy2j4.3
