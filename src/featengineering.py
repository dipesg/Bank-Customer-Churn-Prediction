import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import  logger


class Engineering:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../logs/featengLog.txt", 'a+')
        self.data = pd.read_csv("../data/Churn_Modelling.csv")
        self.data.drop(['CustomerId','RowNumber','Surname'],axis='columns',inplace=True)
        
    def feature_engineering(self):
        self.log_writer.log(self.file_object, "Balancing salary by estimated salary.")
        self.data['BalanceSalaryRatio'] = self.data.Balance/self.data.EstimatedSalary
        sns.boxplot(y='BalanceSalaryRatio',x = 'Exited', hue = 'Exited',data = self.data)
        plt.ylim(-1, 5)
        plt.savefig("../plot/feat_eng/BalanceSalary.png")
        
        self.log_writer.log(self.file_object, "Balancing Tenure by age.")
        self.data['TenureByAge'] = self.data.Tenure/(self.data.Age)
        sns.boxplot(y='TenureByAge',x = 'Exited', hue = 'Exited',data = self.data)
        plt.ylim(-1, 1)
        plt.savefig("../plot/feat_eng/TenureByAge.png")
        
        self.log_writer.log(self.file_object, "Printing Catagorical Variables.")
        for column in self.data:
                if self.data[column].dtypes=='object':
                    print(f'{column}: {self.data[column].unique()}')
        
        self.log_writer.log(self.file_object, "Performing Label Encoding.")
        self.data['Gender'].replace({'Male': 1,'Female': 0},inplace=True)
        
        self.log_writer.log(self.file_object, "Performing OneHot Encoding.")
        self.dataset = pd.get_dummies(data=self.data, columns=['Geography'])
        
        self.log_writer.log(self.file_object, "Performing Normalization.")
        scale_var = ['Tenure','CreditScore','Age','Balance','NumOfProducts','EstimatedSalary']
        scaler = MinMaxScaler()
        self.dataset[scale_var] = scaler.fit_transform(self.dataset[scale_var])
        
        return self.dataset
    
if __name__ == "__main__":
    dataset = Engineering().feature_engineering()
    print(dataset.head())
        
        
        
        