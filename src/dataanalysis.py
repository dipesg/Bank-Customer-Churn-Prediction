import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import src.logger as logger

class Analysis:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../logs/dataanalysisLog.txt", 'a+')
        self.data = pd.read_csv("../data/Churn_Modelling.csv")
        self.data.drop(['CustomerId','RowNumber','Surname'],axis='columns',inplace=True)
        
    def piechart(self):
        self.log_writer.log(self.file_object, "Plotting Pie Chart.")
        labels = 'Exited(Churned)', 'Retained'
        sizes = [self.data.Exited[self.data['Exited']==1].count(), self.data.Exited[self.data['Exited']==0].count()]
        explode = (0, 0.1)
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        plt.title("Proportion of customer churned and retained", size = 20)
        plt.savefig("./plot/PieChart.png")
        
    def cat_barplot(self):
        self.log_writer.log(self.file_object, "Plotting Barplot of Catagorical features.")
        fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
        sns.countplot(x='Geography', hue = 'Exited',data = self.data, ax=axarr[0][0])
        sns.countplot(x='Gender', hue = 'Exited',data = self.data, ax=axarr[0][1])
        sns.countplot(x='HasCrCard', hue = 'Exited',data = self.data, ax=axarr[1][0])
        sns.countplot(x='IsActiveMember', hue = 'Exited',data = self.data, ax=axarr[1][1])
        plt.savefig("./plot/Cat_Barplot.png")
        
    def num_boxplot(self):
        self.log_writer.log(self.file_object, "Plotting Boxplot of Numerical Features.")
        fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
        sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = self.data, ax=axarr[0][0])
        sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = self.data , ax=axarr[0][1])
        sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = self.data, ax=axarr[1][0])
        sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = self.data, ax=axarr[1][1])
        sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = self.data, ax=axarr[2][0])
        sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = self.data, ax=axarr[2][1])
        plt.savefig("./plot/Num_Boxplot.png")
        
    def pred_visualization(self):
        self.log_writer.log(self.file_object, "Doing Customer churn prediction visualization.")
        tenure_churn_no = self.data[self.data.Exited==0].Tenure
        tenure_churn_yes = self.data[self.data.Exited==1].Tenure
        plt.xlabel("tenure")
        plt.ylabel("Number Of Customers")
        plt.title("Customer Churn Prediction Visualiztion")
        plt.hist([tenure_churn_yes, tenure_churn_no], rwidth=0.95, color=['red','green'],label=['Churn=Yes','Churn=No'])
        plt.legend()
        plt.savefig("./plot/pred_visualization.png")
        
if __name__ == "__main__":
    Analysis().piechart()
    Analysis().cat_barplot()
    Analysis().num_boxplot()
    Analysis().pred_visualization()