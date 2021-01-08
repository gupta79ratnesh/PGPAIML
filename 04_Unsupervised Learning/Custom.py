
# coding: utf-8

# In[ ]:


import pandas as pandas
import numpy as numpy
import os
import matplotlib.pyplot as matplot
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import itertools
from IPython.display import Image  
from sklearn import tree
from os import system
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import GridSearchCV
numpy.random.seed(1234)
RandomState = numpy.random.seed(1234)
import warnings
warnings.filterwarnings('ignore')
import scipy.stats as stats
from scipy.stats import chi2_contingency

# In[8]:


class Perform_EDA():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        
    def EDA_Corr(df):
        """This gives output as Covariance matrix and feature wise uniquess i.e how much its statistically
        independent. This is done with default range of corr between +0.5 to -0.6"""
        corr = df.corr()
        index = corr.columns
        Output = []
        for i in range(0,len(index)):
            i = index[i]
            Pos = corr.index[(corr[i] >= 0.5)].tolist()
            No = corr.index[(corr[i] < 0.5) & (corr[i] > -0.6)].tolist()
            Neg = corr.index[(corr[i] <= -0.5)].tolist()
            leng_u = len(No)
            leng_pos = len(Pos)
            leng_neg = len(Neg)
            Out = [i, leng_u, leng_pos, leng_neg, Pos, Neg, No]
            Output.append(Out)
        fig, ax = matplot.subplots(figsize=(20,10))  
        sns.heatmap(corr,annot=True,vmin=-1,vmax=1,cmap='Blues', linewidths=0, ax = ax)
        Output1 = pandas.DataFrame(Output, columns= ['Feature','Uniqueness','Positive rel', 'inverse rel', 'Pos', 'Neg', 'No'])
        return Output1

    def EDA(df):
        """This function creates a dataframe transpose with 5 point summary, Kurtosis, Skewness, IQR, Range
        Total count of missing values, % missing values against the total records"""
        EDA = pandas.DataFrame((df.describe()).T)
        EDA["Kurtosis"] = df.kurtosis()
        EDA["Skewness"] = df.skew()
        EDA["Range"] = EDA['max'] -  EDA['min']
        EDA["IQR"] = EDA['75%'] -  EDA['25%']
        EDA["Missing Values"] = df.shape[0] - EDA["count"]
        print("Total Missing Values = ", EDA['Missing Values'].sum(), "Data Points, Contributing to ", 
              (round(((EDA['Missing Values'].sum())/len(df)),2))*100,"%")
        print("Columns with values as 0\n\n",pandas.Series((EDA.loc[EDA['min'] == 0]).index),'\n')
        indices = EDA[EDA['min'] == 0].index
        print("\nColumns with numnber of Zeros\n")
        for i in range(0,len(indices)):
            j = indices[i]
            print(j,"   =",(df[j].value_counts())[0])
        return EDA
    
    def NaN_treatment_median(df):
        EDA_Summary = Perform_EDA.EDA(df)
        Missing_Value_Columns = pandas.Series((EDA_Summary.loc[EDA_Summary['Missing Values'] != 0]).index)
        len(Missing_Value_Columns)
        for i in range(0,len(Missing_Value_Columns)):
            df[Missing_Value_Columns[i]].fillna(df[Missing_Value_Columns[i]].median(), inplace = True)
        return df
    
    def NaN_treatment_mean(df):
        EDA_Summary = Perform_EDA.EDA(df)
        Missing_Value_Columns = pandas.Series((EDA_Summary.loc[EDA_Summary['Missing Values'] != 0]).index)
        len(Missing_Value_Columns)
        for i in range(0,len(Missing_Value_Columns)):
            df[Missing_Value_Columns[i]].fillna(df[Missing_Value_Columns[i]].mean(), inplace = True)
        return df
    
    def Zero_Values_Treatment_Median(df):
        EDA_Summary = Perform_EDA.EDA(df)
        Zero_Value_Columns = pandas.Series((EDA_Summary.loc[EDA_Summary['min'] == 0]).index)
        for i in range(0,len(Zero_Value_Columns)):
            df[Zero_Value_Columns[i]].replace(0,(df[Zero_Value_Columns[i]]).median(), inplace = True)
        return df
    
    def Zero_Values_Treatment_Mean(df):
        EDA_Summary = Perform_EDA.EDA(df)
        Zero_Value_Columns = pandas.Series((EDA_Summary.loc[EDA_Summary['min'] == 0]).index)
        for i in range(0,len(Zero_Value_Columns)):
            df[Zero_Value_Columns[i]].replace(0,(df[Zero_Value_Columns[i]]).mean(), inplace = True)
        return df

    def Missing_Values(df):
        for i in range(0,len(df.columns)):
            i = df.columns[i]
            print(i,'\n\n',df[i].value_counts(),'\n\n',(df[i].value_counts()).sum(),'\n\n')
        return Missing_Values
    
    def EDA_target(df,Y):
        DF = pandas.DataFrame(Y.value_counts())
        DF['Contribution'] = round(((DF['class'])/len(df)*100),2)
        Missing_values = len(df) - (Y.value_counts()).sum()
        print("Total Missing Values = ", Missing_values, "Data Points, Contributing to ",
              round(((Missing_values/len(df))*100),2),"%")
        return DF
    
    def univariate_plots(Source):
        print("Columns that are int32,int64 = ",Source.select_dtypes(include=['int32','int64']).columns)
        print("Columns that are flaot32,float64 = ",Source.select_dtypes(include=['float64']).columns)
        print("Columns that are objects = ",Source.select_dtypes(include=['object']).columns)
        a = pandas.Series(Source.select_dtypes(include=['int32','int64']).columns)
        leng = len(a)
        for j in range(0,len(a)):
            f, axes = matplot.subplots(1, 2, figsize=(10, 10))
            sns.boxplot(Source[a[j]], ax = axes[0])
            sns.distplot(Source[a[j]], ax = axes[1])
            matplot.subplots_adjust(top =  1.5, right = 10, left = 8, bottom = 1)

        a = pandas.Series(Source.select_dtypes(include=['float64']).columns)
        leng = len(a)
        for j in range(0,len(a)):
            matplot.Text('Figure for float64')
            f, axes = matplot.subplots(1, 2, figsize=(10, 10))
            sns.boxplot(Source[a[j]], ax = axes[0])
            sns.distplot(Source[a[j]], ax = axes[1])
            matplot.subplots_adjust(top =  1.5, right = 10, left = 8, bottom = 1)

        a = pandas.Series(Source.select_dtypes(include=['object']).columns)
        leng = len(a)
        for j in range(0,len(a)):
            matplot.subplots()
            sns.countplot(Source[a[j]])
            
    def Impute_Outliers(df,method = "median",threshold = 0.1):
            """Pls input the method as a string - mean or median with 'm' in lower case
            Default method = median
            Detault threshold = 0.1
            The function will give 3 outputs 
            1. df data imputed based on the value provided
            2. Outlier impact as a printed message with % of records impacted
            """
            df_Columns = df.columns
            Subset_Columns = pandas.Series(df.select_dtypes(include=['int32','int64','float64','float32']).columns)

            Subset = df[Subset_Columns]

            IQR = Subset.quantile(0.75) - Subset.quantile(0.25)

            Q3_values = Subset.quantile(0.75) + (1.5 * IQR)
            Q1_values = Subset.quantile(0.25) - (1.5 * IQR)

            Q1 = []
            for i in range(1,len(Subset_Columns)+1):
                c = "Q1"+str(i)
                Q1.append(c)

            Q3 = []
            for i in range(1,len(Subset_Columns)+1):
                c = "Q3"+str(i)
                Q3.append(c)

            df[Q3] = Subset > Q3_values[0:len(Subset_Columns)]
            df[Q1] = Subset < Q1_values[0:len(Subset_Columns)]

            Q1_Outliers = []
            Q1_j = []
            Q3_Outliers = []
            Q3_j = []
            for i in range(0,len(Q1)):
                i = Q1[i]
                No = df.shape[0] - df[i].value_counts()[0]
                Q1_Outliers.append(No)
                Q1_j.append(i)
            Q1_Col = pandas.DataFrame(Q1_j, columns=["Q1"])
            Q1_outliers = pandas.DataFrame(Q1_Outliers, columns=["Q1 Outliers"])
            Outliers_impact_Q1 = Q1_Col.join(Q1_outliers)

            for i in range(0,len(Q3)):
                i = Q3[i]
                No = df.shape[0] - df[i].value_counts()[0]
                Q3_Outliers.append(No)
                Q3_j.append(i)
            Q3_Col = pandas.DataFrame(Q3_j, columns=["Q3"])
            Q3_outliers = pandas.DataFrame(Q3_Outliers, columns=["Q3 Outliers"])
            Outliers_impact_Q3 = Q3_Col.join(Q3_outliers)

            Outliers_impact = Outliers_impact_Q1['Q1 Outliers']+Outliers_impact_Q3['Q3 Outliers']
            Outliers_impact = (pandas.DataFrame(Subset_Columns, columns=["Column Name"])).join(pandas.DataFrame(Outliers_impact, columns=["No of Outliers"]))
            print(Outliers_impact)


            aij = []
            for i in range(0,len(Q3)):
                i = Q3[i]
                bij = ((pandas.DataFrame(df[i])).index[(df[i] == True)].tolist())
                aij = aij + bij
            Q3_indices = (pandas.Series(aij)).value_counts()


            cij = []
            for i in range(0,len(Q1)):
                i = Q1[i]
                dij = ((pandas.DataFrame(df[i])).index[(df[i] == True)].tolist())
                cij = cij + dij
            Q1_indices = (pandas.Series(cij)).value_counts()

            print("No of records impacted by Outliers = ",round((Outliers_impact['No of Outliers'].sum() / len(df)),2)*100,"%")
            print("No of records in outliers beyond Q4 = ",round(((pandas.DataFrame(Q3_Outliers)[0]).sum() / len(df)),2)*100,"%")
            print("No of records in outliers beyond Q1 = ",round(((pandas.DataFrame(Q1_Outliers)[0]).sum() / len(df)),2)*100,"%")

            if (round((Outliers_impact['No of Outliers'].sum() / len(df)),2)) <= threshold :
                print((round((Outliers_impact['No of Outliers'].sum() / len(df)),2)*100)," ",threshold)
                Outliers_Q3_Q1 = pandas.DataFrame(Q3_values, columns = ['Q3_values']).join(pandas.DataFrame(Q1_values, columns=['Q1_values']))
                for i in range(0,len(Subset_Columns)):
                    Q3 = ((Outliers_Q3_Q1).T)[Subset_Columns[i]].loc['Q3_values']
                    Q1 = ((Outliers_Q3_Q1).T)[Subset_Columns[i]].loc['Q1_values']
                    df.loc[df[Subset_Columns[i]] > Q3, Subset_Columns[i]] = numpy.nan
                    df.loc[df[Subset_Columns[i]] < Q1, Subset_Columns[i]] = numpy.nan
                    if method == "median":
                        median1 = ((df.loc[(df[Subset_Columns[i]]<((((Outliers_Q3_Q1).T)[Subset_Columns[i]])['Q3_values'])) & 
                         (df[Subset_Columns[i]]>((((Outliers_Q3_Q1).T)[Subset_Columns[i]])['Q1_values']))])[Subset_Columns[i]]).median()
                    else:
                        median1 = ((df.loc[(df[Subset_Columns[i]]<((((Outliers_Q3_Q1).T)[Subset_Columns[i]])['Q3_values'])) & 
                         (df[Subset_Columns[i]]>((((Outliers_Q3_Q1).T)[Subset_Columns[i]])['Q1_values']))])[Subset_Columns[i]]).mean()
                df.replace(numpy.nan,median1,inplace= True)
                print("No of records imputed using the",method,"is",Outliers_impact['No of Outliers'].sum())
                df = df.iloc[:,0:len(df_Columns)]
                return df    
            else:
                print((round((Outliers_impact['No of Outliers'].sum() / len(df)),2))," ",threshold)
                print("Too many outliers, please alter the 'threshold' if outliers will have to be treated")
                df = df.iloc[:,0:len(df_Columns)]
                return df
        
class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX,alpha):
        result = ""
        if self.p<alpha:
            result="{0} exhibits multicollinearity. (Consider Discarding {0} from model)".format(colX)
        else:
            result="{0} is NOT related and can be a good predictor".format(colX)

        print(result)
        
    def TestIndependence(self,colX,colY,alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pandas.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pandas.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        #print(self.dfExpected)
        self._print_chisquare_result(colX,alpha)
        return self.dfExpected

