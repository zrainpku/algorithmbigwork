import numpy as np
from sklearn import preprocessing
enc=preprocessing.OneHotEncoder()
enc.fit([['a',0,3],['1',1,0],['0',2,1],['1',0,2]])
list=enc.transform([[0,1,1],[1,1,0]]).toarray()
list=np.array(list,dtype=int)
print list



