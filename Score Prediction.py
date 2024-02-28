#!/usr/bin/env python
# coding: utf-8

# In[154]:


import pandas as pd
import numpy as np


# In[155]:


match=pd.read_csv('matches.csv')
delivery=pd.read_csv('deliveries.csv')


# In[156]:


match.head()


# In[157]:


match.shape


# In[158]:


delivery.head()


# In[159]:


total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()


# In[160]:


total_score_df = total_score_df[total_score_df['inning'] == 1]


# In[161]:


total_score_df


# In[162]:


match_df = match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')


# In[163]:


match_df


# In[164]:


match_df['team1'].unique()


# In[165]:


teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]


# In[166]:


match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')


# In[167]:


match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]


# In[168]:


match_df.shape


# In[169]:


match_df = match_df[match_df['dl_applied'] == 0]


# In[170]:


match_df = match_df[['match_id','city','winner','total_runs']]


# In[171]:


delivery_df = match_df.merge(delivery,on='match_id')


# In[172]:


delivery_df = delivery_df[delivery_df['inning'] == 2]


# In[173]:


delivery_df .shape


# In[174]:


delivery_df['balls+left'] =126 -(delivery_df['over']*6 +delivery_df['ball'])


# In[175]:


delivery_df


# In[176]:


delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].transform(lambda x: x.cumsum())


# In[177]:


delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']


# In[178]:


delivery_df['balls_left'] = 126 - (delivery_df['over']*6 + delivery_df['ball'])


# In[179]:


delivery_df


# In[180]:


delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna(0)
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: 0 if x == 0 else 1)
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
delivery_df['wickets'] = delivery_df.groupby('match_id')['player_dismissed'].cumsum()
delivery_df['wickets'] = 10 - delivery_df['wickets']
delivery_df.head()


# In[181]:


delivery_df.tail()


# In[182]:


delivery_df['crr'] = (delivery_df['current_score']*6)/(120 - delivery_df['balls_left'])


# In[183]:


delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']


# In[184]:


def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0


# In[185]:


delivery_df['result'] = delivery_df.apply(result,axis=1)


# In[186]:


final_df =delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result' ]]


# In[187]:


final_df = final_df.sample(final_df.shape[0])


# In[188]:


final_df.sample()


# In[189]:


final_df.dropna(inplace=True)


# In[190]:


final_df = final_df[final_df['balls_left'] != 0]


# In[191]:


X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[192]:


X_train


# In[193]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')


# In[194]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[195]:


pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])


# In[196]:


pipe.fit(X_train,y_train)


# In[197]:


y_pred = pipe.predict(X_test)


# In[198]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[199]:


pipe.predict_proba(X_test)[10]


# In[200]:


def match_summary(row):
    print("Batting Team-" + row['batting_team'] + " | Bowling Team-" + row['bowling_team'] + " | Target- " + str(row['total_runs_x']))
    


# In[201]:


def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target
    


# In[202]:


temp_df,target = match_progression(delivery_df,74,pipe)
temp_df


# In[203]:


import matplotlib.pyplot as plt
plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
plt.title('Target-' + str(target))


# In[204]:


teams


# In[205]:


delivery_df['city'].unique()


# In[206]:


import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[ ]:




