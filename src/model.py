import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns 
import matplotlib.pyplot as plt
import logger
import pickle
import pandas as pd
from featengineering import Engineering
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

class Model:
    def __self__(self):
        self.data = Engineering().feature_engineering()
        self.X = self.data.drop('Exited',axis='columns')  ##independent features
        self.y = self.data['Exited']  ##dependent feature
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=5)
        
    def model_building(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../logs/ModelBuildingLog.txt", 'a+')
        self.data = Engineering().feature_engineering()
        self.X = self.data.drop('Exited',axis='columns')  ##independent features
        self.y = self.data['Exited']  ##dependent feature
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=5)
        self.log_writer.log(self.file_object, "Building ANN model.")
        model = keras.Sequential([
            keras.layers.Dense(32, input_shape=self.x_train.shape, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        self.log_writer.log(self.file_object, "Compiling Model.")
        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
        self.log_writer.log(self.file_object, "Fitting Model.")
        model.fit(self.x_train, self.y_train, epochs=120)
        
        self.log_writer.log(self.file_object, "Evaluating Model.")
        model.evaluate(self.x_test, self.y_test)
        
        self.log_writer.log(self.file_object, "Now Predicting.")
        prediction = model.predict(self.x_test)
        
        self.log_writer.log(self.file_object, "Converting our prediction to 0 and 1 to check accuracy.")
        y_pred = []
        for element in prediction:
            if element > 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
        
        self.log_writer.log(self.file_object, "Checking the accuracy..")        
        print(classification_report(self.y_test,y_pred))
        
        self.log_writer.log(self.file_object, "Plotting our prediction.") 
        cm = tf.math.confusion_matrix(labels=self.y_test,predictions=y_pred)
        plt.figure(figsize = (10,7))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.savefig("../plot/training/training.png")
        
        print("Accuracy score is: ", accuracy_score(self.y_test,y_pred)*100,"%")
        
        model.save("model.h5")
        
    def random_forest(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../logs/ModelBuildingLog.txt", 'a+')
        self.data = Engineering().feature_engineering()
        self.X = self.data.drop('Exited',axis='columns')  ##independent features
        self.y = self.data['Exited']  ##dependent feature
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=5)
        self.log_writer.log(self.file_object, "Building random forest  model.")
        
        self.model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=100)
        self.model.fit(self.x_train,self.y_train)
        self.pred = self.model.predict(self.x_test)
        self.score = accuracy_score(self.y_test, self.pred)
        print(self.score*100,"%")
        pickle.dump(self.model, open('model.pkl', 'wb'))
        
    def hypertune_randomforest(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../logs/ModelBuildingLog.txt", 'a+')
        self.data = Engineering().feature_engineering()
        self.X = self.data.drop('Exited',axis='columns')  ##independent features
        self.y = self.data['Exited']  ##dependent feature
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=5)
        self.log_writer.log(self.file_object, "Tuning Random Forest Model.")
        param_distributions = {"n_estimators": [1, 2, 5, 10, 20, 50, 100, 200, 500],"max_leaf_nodes": [2, 5, 10, 20, 50, 100]}
        search1_cv = RandomizedSearchCV(
            RandomForestClassifier(), param_distributions=param_distributions,
            scoring="accuracy", n_iter=10, random_state=0, n_jobs=2,
        )
        search1_cv.fit(self.x_train, self.y_train)
        #print(f"Best score: {search_cv.score(x_test,y_test)}")

        columns = [f"param_{name}" for name in param_distributions.keys()]
        columns += ["mean_test_error", "std_test_error"]
        cv_results = pd.DataFrame(search1_cv.cv_results_)
        cv_results["mean_test_error"] = -cv_results["mean_test_score"]
        cv_results["std_test_error"] = cv_results["std_test_score"]
        cv_results[columns].sort_values(by="mean_test_error")
        
        # Findinding best tuned parameter
        print(search1_cv.best_params_)
        
        
        
if __name__ == "__main__":
    Model().random_forest()

        
        