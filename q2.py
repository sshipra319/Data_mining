import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  #library for TFidf weight matrix

u1 = [1,0]
u2 = [0,1]
sigma1 = [[1, 0.75],[0.75, 1]]
sigma2 = [[1, 0.75],[0.75, 1]]

class NaiveBayes:  
    
    def data_calc(self, values):
        label0_train = np.zeros(values).T
        label1_train = np.ones(values).T
        labels_trained = np.append(label0_train,label1_train,axis=0)
        label0_test = np.zeros(500).T
        label1_test = np.ones(500).T
        labels_tested = np.append(label0_test,label1_test,axis=0)
        trainset1 = np.random.multivariate_normal(u1, sigma1, values)
        trainset2 = np.random.multivariate_normal(u2, sigma2, values)
        traindata = np.append(trainset1, trainset2, axis=0)
        testset1 = np.random.multivariate_normal(u1, sigma1, 500)
        testset2 = np.random.multivariate_normal(u2, sigma2, 500)
        testdata = np.append(testset1,testset2, axis=0)
        return (traindata, testdata, labels_trained, labels_tested)
    
    def myNB(self,X,Y,X_test,Y_test):
        df=pd.DataFrame((X),columns=['X','Y'])
        df['label']=Y        

        label_0=df[df['label']==0]
        label_1=df[df['label']==1]
        
        prior_0=len(label_0)/(len(label_0)+len(label_1))
        prior_1=len(label_1)/(len(label_0)+len(label_1))
        
        label_0mean = label_0.mean()
        label_1mean = label_1.mean()
        label_0std = label_0.std()
        label_1std = label_1.std()        

        class0=[]
        class1=[]
        df_test=pd.DataFrame((X_test),columns=['X','Y'])
        for rows in df_test.iterrows():
            class0.append((ss.norm.pdf(rows[1]['X'],label_0mean['X'],label_0std['X']))*(ss.norm.pdf(rows[1]['Y'],label_0mean['Y'],label_0std['Y']))*prior_0)
            class1.append((ss.norm.pdf(rows[1]['X'],label_1mean['X'],label_1std['X']))*(ss.norm.pdf(rows[1]['Y'],label_1mean['Y'],label_1std['Y'])*prior_1))
        df_test['class_0_prob']=class0
        df_test['class_1_prob']=class1
        #print (df_test)      

        prediction=[]
        for rows in df_test.iterrows():
            if rows[1]['class_0_prob']>rows[1]['class_1_prob']:
                prediction.append(0)
            else:
                prediction.append(1)
                
        df_test['predict']=prediction
        
        df_test['actual']=Y_test        

        count=0
        
        for rows in df_test.iterrows():
            if rows[1]['predict']==rows[1]['actual']:
                count+=1
                
        accuracy=count/(df_test.shape[0])
        error=1-accuracy
        tp=0
        fp=0
        tn=0
        fn=0
        class_0_x=[]
        class_0_y=[]
        class_1_x=[]
        class_1_y=[]
        for rows in df_test.iterrows():
            if rows[1]['predict']==rows[1]['actual']:
                if rows[1]['predict']==1:
                    tp+=1
                else:
                    tn+=1
            elif rows[1]['predict']==1 and rows[1]['actual']==0:
                fp+=1
            else:
                fn+=1
            if rows[1]['predict']==0:
                class_0_x.append(rows[1]['X'])
                class_0_y.append(rows[1]['Y'])
            else:
                class_1_x.append(rows[1]['X'])
                class_1_y.append(rows[1]['Y'])
        plt.title('ScatterPlot of labeled Data')
        plt.scatter(class_0_x,class_0_y,1,'green', label = 'Label 0')        
        plt.scatter(class_1_x,class_1_y,1,'red' , label = 'Lable 1')
        plt.show()      

        accuracy=((tp+tn)/(tp+tn+fp+fn))*100
        print('Average Accuracy (%): ',accuracy)
        
        error = 100 - accuracy
        print('Average Error Rate: ' , error)       

        recall=(tp/(tp+fn))*100
        print('Average Recall:',recall)        

        precision=(tp/(tp+fp))*100
        print('Average Precision:',precision)        

        conf_matrix=pd.DataFrame([[tp,fn],[fp,tn]],columns=['Actual 1','Actual 0'])
        print('Confusion matrix\n:',conf_matrix)   
        
        po = pd.DataFrame(class0,columns=['class_0_prob'])
        po['class_1_prob'] = class1
        return (prediction,po,error)
    
    def part_1_1(self):
        print('Part 1_1')
        a,b,c,d= self.data_calc(500)
        pred,posterior,err= self.myNB(a,c,b,d)
        frame = pd.DataFrame(posterior, columns=['class_0_prob','class_1_prob'])
        frame['actual']=d
        self.calc_ROC_curve(frame)    
    
    def part_1_2(self):
        print('Part 1_2')
        data_list=[10,20,50,100,300,500]
        accuracy=[]
        for i in range (len(data_list)):
            a,b,c,d=self.data_calc(data_list[i])
            pred,posterior,err=self.myNB(a,c,b,d)
            accuracy.append(1-err)
        plt.title('Changes of accuracies')
        plt.plot(data_list,accuracy)
        plt.show()
    
    def part_1_3_data_calc(self):
        label0_train = np.zeros(700).T
        label1_train = np.ones(300).T
        labels_trained =np.append(label0_train,label1_train,axis=0)
        label0_test = np.zeros(500).T
        label1_test = np.ones(500).T
        labels_tested =np.append(label0_test,label1_test,axis=0)
        trainset1 = np.random.multivariate_normal(u1, sigma1, 700)
        trainset2 = np.random.multivariate_normal(u2, sigma2, 300)
        traindata = np.append(trainset1, trainset2, axis=0)
        testset1 = np.random.multivariate_normal(u1, sigma1, 500)
        testset2 = np.random.multivariate_normal(u2, sigma2, 500)
        testdata = np.append(testset1,testset2, axis=0)
        return (traindata, testdata, labels_trained, labels_tested)
    
    def part_1_3(self):
        print('Part 1_3')
        a,b,c,d= self.part_1_3_data_calc()
        pred,posterior,err= self.myNB(a,c,b,d)        
        print(type(posterior))
        frame = pd.DataFrame(posterior, columns=['class_1_prob'])
        frame['actual']=d
        self.calc_ROC_curve(frame)
        
    def calc_ROC_curve(self,df_test):
        print('Part 2')
        tp=0
        fp=0
        tpr=[]
        fpr=[]
        actual_p=0
        actual_n=0
        fpr_prv=0
        auc=0
        df_test=df_test.sort_values(by='class_1_prob', ascending=False)
        for rows in df_test.iterrows():
            if rows[1]['actual']==1:
                actual_p+=1
            else:
                actual_n+=1
        for rows in df_test.iterrows():
                if rows[1]['actual']==1:
                    tp+=1
                else: 
                    fp+=1
                tpr.append((tp/(actual_p)))
                fpr.append((fp/(actual_n)))
                auc+=((tp/actual_p))*((fp/actual_n)-fpr_prv)
                fpr_prv=(fp/actual_n)
        print('Area Under Curve:',auc)
                
        plt.title('ROC Curve')
        plt.plot(fpr,tpr)
        plt.show()
    
    def weightMatrix(self):
        Amazon_Reviews = pd.read_csv('Amazon_Reviews.csv')  #reading the data file
        Amazon_Reviews1 = Amazon_Reviews['Review']  #selecting only the review column   it has Y test as labels in the file
        Review_list = Amazon_Reviews1.values.tolist()  #coverting data to list
        YTest_amzon = Amazon_Reviews['Label']         
        YTest_amzon_List = YTest_amzon.values.tolist()  #coverting data to list
        print(YTest_amzon_List)
        for i in range(len(YTest_amzon_List)):
            if YTest_amzon_List[i] == '__label__2 ':
                YTest_amzon_List[i] = 1
            else:
                YTest_amzon_List[i] = 0
        
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(Review_list)
        df1 = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names()) #dataframes using panda
        
        df_array = df1.to_numpy()
        print(df_array)
        

def main():
    nb = NaiveBayes()
    nb.part_1_1()
    nb.part_1_2()
    nb.part_1_3()
    nb.weightMatrix()
main()