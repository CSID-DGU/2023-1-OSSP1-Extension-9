import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import scipy.stats as s
import matplotlib.pyplot as plt


class OpenMax:
        
    def fit(self,X,Y):#X는 모델의 predict값 List Y는 원본라벨값 List
        class_result = []
        for a,b,c,d in X:
            numbers = [a,b,c,d]
            class_result.append(numbers.index(max(numbers)))
    
        tf_result = (Y == class_result)

        for_open_max_0 = []
        for_open_max_1 = []
        for_open_max_2 = []
        for_open_max_3 = []
        for i in range(len(class_result)):
            if(tf_result[i] == True):
                if(class_result[i] == 0):
                    for_open_max_0.append(X[i])
                if(class_result[i] == 1):
                    for_open_max_1.append(X[i])
                if(class_result[i] == 2):
                    for_open_max_2.append(X[i])
                if(class_result[i] == 3):
                    for_open_max_3.append(X[i])
    
        
        def average_vector(for_open_max, class_result):
            a1=0
            a2=0
            a3=0
            a4=0
            for i in for_open_max:
                a1 += i[0]
                a2 += i[1]
                a3 += i[2]
                a4 += i[3]
            length = len(class_result)
            average = [a1/length, a2/length, a3/length, a4/length]
            return average

        self.average_0 = average_vector(for_open_max_0, class_result) #평균 Logit Vector - Class 0
        self.average_1 = average_vector(for_open_max_1, class_result) #평균 Logit Vector - Class 1
        self.average_2 = average_vector(for_open_max_2, class_result) #평균 Logit Vector - Class 2
        self.average_3 = average_vector(for_open_max_3, class_result) #평균 Logit Vector - Class 3
        
        def distance(for_open_max, average):
            dist = []
            for i in for_open_max:
                #m = i - average
                m = np.subtract(i,average)
                distance = (m[0]**2) + (m[1]**2) + (m[2]**2) + (m[3]**2)
                dist.append(distance)
            return dist

        self.maxdist0 = distance(for_open_max_0, self.average_0)
        self.maxdist1 = distance(for_open_max_1, self.average_1)
        self.maxdist2 = distance(for_open_max_2, self.average_2)
        self.maxdist3 = distance(for_open_max_3, self.average_3)

        self.maxdist0.sort(reverse =True)
        self.maxdist1.sort(reverse =True)
        self.maxdist2.sort(reverse =True)
        self.maxdist3.sort(reverse =True)
        
        
        
    def calculate(self,logits):#모델의 predict값을 입력으로 받음 [1,2,3,4]
        def distance(logit_vector, average):
            m = []
            for i in range(0,4):
                m.append(logit_vector[i] - average[i])
            distance = (m[0]**2) + (m[1]**2) + (m[2]**2) + (m[3]**2)
            return distance

        dist0 = distance(logits, self.average_0)
        dist1 = distance(logits, self.average_1)
        dist2 = distance(logits, self.average_2)
        dist3 = distance(logits, self.average_3)

        def calculCDF(dist, input_loc, input_scale):
            CDF = s.exponweib.cdf(dist, *s.exponweib.fit(dist, 2, 5, scale=input_scale, loc=input_loc))     
            return CDF

        def weib(x,n,a):
            return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)
        
        md0 = self.maxdist0.copy()
        md1 = self.maxdist1.copy()
        md2 = self.maxdist2.copy()
        md3 = self.maxdist3.copy()
        

        md0.insert(20,dist0)
        md0 = md0[0:21]
        md0.sort(reverse=True)
        index0 = md0.index(dist0)
        
        md1.insert(20,dist1)
        md1 = md1[0:21]
        md1.sort(reverse=True)
        index1 = md1.index(dist1)
        
        md2.insert(20,dist2)
        md2 = md2[0:21]
        md2.sort(reverse=True)
        index2 = md2.index(dist2)
        
        md3.insert(20,dist3)
        md3 = md3[0:21]
        md3.sort(reverse=True)
        index3 = md3.index(dist3)

        CDF0 = s.exponweib.cdf(md0, *s.exponweib.fit(md0, 1, 1, scale=2, loc=0))
        CDF1 = s.exponweib.cdf(md1, *s.exponweib.fit(md1, 1, 1, scale=2, loc=0))
        CDF2 = s.exponweib.cdf(md2, *s.exponweib.fit(md2, 1, 1, scale=2, loc=0))
        CDF3 = s.exponweib.cdf(md3, *s.exponweib.fit(md3, 1, 1, scale=2, loc=0))

        input_CDF0 = CDF0[index0]
        input_CDF1 = CDF1[index1]
        input_CDF2 = CDF2[index2]
        input_CDF3 = CDF3[index3]
        
        
        updated_logit = []

        logit0 = logits[0]-(logits[0]*input_CDF0)
        logit1 = logits[1]-(logits[1]*input_CDF1)
        logit2 = logits[2]-(logits[2]*input_CDF2)
        logit3 = logits[3]-(logits[3]*input_CDF3)
        unknown_logit = (input_CDF0*logits[0]) + (input_CDF1*logits[1]) + (input_CDF2*logits[2]) + (input_CDF3*logits[3]) # unknown class의 logit vector     

        updated_logit = [logit0, logit1, logit2, logit3, unknown_logit]

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        
        result_arr = softmax(updated_logit)
        arr = result_arr.tolist()
        return [arr[0],arr[1],arr[2],arr[3] + arr[4]]#[혐오성,성차별,일베,Unknown]에 해당하는 확률

