import pandas as pd 
import numpy as np

##load data and preproccess
data = pd.read_csv("nursery.csv").sample(frac=1)
_len=int(data.shape[0]*0.7)
train_target=data['final evaluation'].iloc[:_len]
train=data.drop(['final evaluation'],axis=1).iloc[:_len]
test_target=data['final evaluation'].iloc[_len:]
test=data.drop(['final evaluation'],axis=1)[_len:]

def decision_tree(data,target,max_deep=8):
  if data.empty or max_deep==0:
    return target.mode()[0]
  ##### Total Entropy
  Total_Entropy=Entropy(target)
  ##### find feature with  most IG
  max_IG=-1
  prefered_feature=''
  for c in data.columns:
    p=data[c].value_counts(normalize=True)
    total_E=0
    for x in p.index:
      E=Entropy(target[data[c]==x])
      total_E+=p[x]*E
    if Total_Entropy-total_E>max_IG:
      max_IG=Total_Entropy-total_E
      prefered_feature=c
  ##### make tree
  tree={}
  for child in data[prefered_feature].unique():
    _data=data[data[prefered_feature]==child].drop([prefered_feature],axis=1)
    _target=target[data[prefered_feature]==child]
    sub_tree=decision_tree(_data,_target,max_deep-1)
    if sub_tree!={}:
      tree[prefered_feature+' : '+child]=sub_tree
  #### haras mikonim
  if list(tree.values()).count(list(tree.values())[0]) == len(tree):
    return list(tree.values())[0]
  return tree
  
def Entropy(target):
    p=target.value_counts(normalize=True)
    return (-p*np.log2(p)).sum()

def predict(tree,test_data):
  output=pd.DataFrame([],columns=['target'],index=test_data.index)
  if (type(tree) is not dict):
    output['target']=tree
    return output
  for child in tree:
    feature=child.split(' : ')[0]
    selected=child.split(' : ')[1]
    if not test_data[test_data[feature]==selected].empty:
      out=predict(tree[child],test_data[test_data[feature]==selected])
      output.loc[out.index,'target']=out['target']
  return output
  
ID3=decision_tree(train,train_target,5)
y_predicted=predict(ID3,test)

