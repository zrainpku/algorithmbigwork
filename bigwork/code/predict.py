import numpy as np
import scipy as sp
import math
from sklearn.cross_validation import train_test_split
from utils import load_train_data,load_test_data
import os
import xgboost as xgb
import csv
from IPython import embed
# def write_pred(file_path,list_pred):
#     fw=open(file_path,'w')
#     for i in list_pred:
#         fw.write(str(i)+'\n')

def MSE(list1,list2):
    len_list1=len(list1)
    len_list2=len(list2)
    if not len_list1==len_list2:
        return 0
    sum=0
    for i in range(len_list1):
        sum+=(list1[i]-list2[i])**2
    result=math.sqrt(sum/len_list1)
    return result

if __name__=='__main__':
    fuse_list,id_list,label_list,len_digit=load_train_data()
    x_train,x_test,y_train,y_test=train_test_split(fuse_list,label_list,test_size=0.2)
    xg_train=xgb.DMatrix(x_train,label=y_train)
    xg_test=xgb.DMatrix(x_test,label=y_test)

    param={
        'eta':0.3,
        'max_depth':4,
        'gamma':1.0,
        'min_child_weight':1,
        'subsample':0.8,
        'save_period':0,
        'booster':'gbtree',
        #'silent':1,
        'nthread':4,
        'objective':'count:poisson',
        #'eval_metric':'logloss'
    }

    watchlist=[(xg_train,'train'),(xg_test,'test')]
    num_round=100
    bst=xgb.train(param,xg_train,num_boost_round=num_round,evals=watchlist)
    pred=bst.predict(xg_test)
    #regression problem does not need to compute error rate.
    #error_rate=sum(int(pred[i])!=y_test[i] for i in range(len(y_test)))/float(len(y_test))
    #print 'error rate is: '+str(error_rate)
    print 'predict done! '
    mse=MSE(pred,y_test)
    print 'mean squared error is: '+str(mse)
    print 'now computing test_set : '
    fuse_list_test,id_list_test,len_digit=load_test_data()
    fuse_list_test=xgb.DMatrix(fuse_list_test)
    pred_test=bst.predict(fuse_list_test)
    csv_result='../result/test_result.csv'
    csvfile=file(csv_result,'wb')
    writer=csv.writer(csvfile)
    writer.writerow(['Id','Score'])
    data=[]
    for i in range(len(id_list_test)):
        id_tmp=id_list_test[i]
        score_tmp=pred_test[i]
        array_tmp=(id_tmp,score_tmp)
        data.append(array_tmp)
    writer.writerows(data)
    csvfile.close()
    if os.path.exists('../result/test_result.csv'):
        print 'save done!'

