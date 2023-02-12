#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# 1. HDI-Calculating three indexes

# 1.1 Education index-Mean Years of Schooling

# 1.1.1 Mean Years of Schooling Index (MYSI): SSP1-3

# In[3]:


pd.options.display.max_columns = 30
pd.options.display.max_rows = 30


# In[2]:


population123 = pd.read_csv(r'D:\HDI-Python\WCDE Data\population size (SSP1-3).csv')

#drop 'World'
population123 


# In[7]:


education123 = pd.read_csv(r'D:\HDI-Python\WCDE Data\mean years of schooling (SSP1-3).csv')

# delete column : age =15+
education123 = education123.drop(list(education123)[3],axis = 1)

education123.loc[:, 'MYSI'] = education123['Years']/17


education123


# In[15]:


#long to wide
education1=education123[0:6054]
education1=education1.drop(columns=['Scenario'])
education1.rename(columns={'Years':'SSP1Years','MYSI':'SSP1MYSI'},inplace = True)

education2=education123[6054:12108]
education2=education2.drop(columns=['Scenario'])
education2.rename(columns={'Years':'SSP2Years','MYSI':'SSP2MYSI'},inplace = True)

education3=education123[12108:18162]
education3=education3.drop(columns=['Scenario'])
education3.rename(columns={'Years':'SSP3Years','MYSI':'SSP3MYSI'},inplace = True)

education1
education2
education3

#Transformation of education123 
#can only merge two at a time

education12 = pd.merge(education1,education2,on=['Year','ISOCode','Area'])
education12

education321 = pd.merge(education12,education3,on=['Year','ISOCode','Area'])
education321


# 1.1.2 Mean Years of Schooling Index (MYSI): SSP4-5

# 1.1.2.1 SSP4 MYSI calculation

# In[16]:


#population
SSP4epop_wide = pd.read_csv(r'D:\HDI-Python\WCDE Data\Global population and human capital projections\SSP4epop_wide.csv')

SSP4epop_wide["sexno"].value_counts()

# sex = both (sexno = 0); education = total (eduno = 0)； ageno>4 (population-age15+)
SSP4epop_wide = SSP4epop_wide.drop(SSP4epop_wide[(SSP4epop_wide['sexno']>0)].index)
SSP4epop_wide = SSP4epop_wide.drop(SSP4epop_wide[(SSP4epop_wide['eduno']>0)].index)
SSP4epop_wide = SSP4epop_wide.drop(SSP4epop_wide.columns[[4,5,6,7]], axis = 1)

#drop=true
SSP4epop_wide = SSP4epop_wide.reset_index(drop=True)

# all population
SSP4epop_wide.loc[:,'pop15']= SSP4epop_wide.iloc[:,4:22].sum(axis=1)

SSP4epop_wide


# In[17]:


#mean years of schooling
SSP4mys_wide = pd.read_csv(r'D:\HDI-Python\WCDE Data\Global population and human capital projections\SSP4mys_wide.csv')
SSP4mys_wide = SSP4mys_wide.drop(SSP4mys_wide[(SSP4mys_wide['sexno']>0)].index)
# sex = both (sexno = 0)

SSP4mys_wide = SSP4mys_wide.reset_index(drop=True)

SSP4mys_wide


# In[18]:


#MYSI-SSP4

#MYS * population

i = SSP4mys_wide.filter(like='ageno_')
j = SSP4epop_wide.filter(like='ageno_')

SSP4mys_wide['shooling'] = np.einsum('ij,ij->i',i,j)  # (i.values * j).sum(axis=1)

#MYS
SSP4mys_wide.loc[:,"Years"] = SSP4mys_wide['shooling']/SSP4epop_wide['pop15']

#MYSI
SSP4mys_wide.loc[:,"MYSI"] = SSP4mys_wide['Years']/17


SSP4mys_wide


# In[19]:


# New dataframe - SSP4 MYSI

SSP4MYSI = SSP4mys_wide.drop(list(SSP4mys_wide)[2:23],axis = 1)


#SSP4MYSI = SSP4MYSI.reset_index(drop=True)

#inset scenario SSP4

#SSP4MYSI.insert (loc = 0, column = 'Scenario', value = "SSP4")
SSP4MYSI.rename(columns={'year':'Year','Years':'SSP4Years','isono':'ISOCode','MYSI':'SSP4MYSI'},inplace = True)

SSP4MYSI


# 1.1.2.2 SSP5 MYSI calculation

# In[20]:


#population
SSP5epop_wide = pd.read_csv(r'D:\HDI-Python\WCDE Data\Global population and human capital projections\SSP5epop_wide.csv')

SSP5epop_wide["sexno"].value_counts()

# sex = both (sexno = 0); education = total (eduno = 0)； ageno>4
SSP5epop_wide = SSP5epop_wide.drop(SSP5epop_wide[(SSP5epop_wide['sexno']>0)].index)
SSP5epop_wide = SSP5epop_wide.drop(SSP5epop_wide[(SSP5epop_wide['eduno']>0)].index)
SSP5epop_wide = SSP5epop_wide.drop(SSP5epop_wide.columns[[4,5,6,7]], axis = 1)

#drop=true
SSP5epop_wide = SSP5epop_wide.reset_index(drop=True)

# all population
SSP5epop_wide.loc[:,'pop']= SSP5epop_wide.iloc[:,4:22].sum(axis=1)


SSP5epop_wide


# In[21]:


#mean years of schooling
SSP5mys_wide = pd.read_csv(r'D:\HDI-Python\WCDE Data\Global population and human capital projections\SSP5mys_wide.csv')
SSP5mys_wide = SSP5mys_wide.drop(SSP5mys_wide[(SSP5mys_wide['sexno']>0)].index)
# sex = both (sexno = 0)

SSP5mys_wide = SSP5mys_wide.reset_index(drop=True)

SSP5mys_wide


# In[22]:


#MYS-SSP5

#MYS * population

i = SSP5mys_wide.filter(like='ageno_')
j = SSP5epop_wide.filter(like='ageno_')

SSP5mys_wide['shooling'] = np.einsum('ij,ij->i',i,j)  # (i.values * j).sum(axis=1)

#MYS
SSP5mys_wide.loc[:,"Years"] = SSP5mys_wide['shooling']/SSP5epop_wide['pop']

#MYSI
SSP5mys_wide.loc[:,"MYSI"] = SSP5mys_wide['Years']/17

SSP5mys_wide


# In[23]:


# New dataframe - SSP5 MYSI

SSP5MYSI = SSP5mys_wide.drop(list(SSP5mys_wide)[2:23],axis = 1)


#SSP5MYSI = SSP5MYSI.reset_index(drop=True)

#inset scenario SSP5
#SSP5MYSI.insert (loc = 0, column = 'Scenario', value = "SSP5")

SSP5MYSI.rename(columns={'year':'Year','Years':'SSP5Years','isono':'ISOCode','MYSI':'SSP5MYSI'},inplace = True)

SSP5MYSI

#d = SSP4MYSI["MYS"].describe()
#d

#filter the data
SSP5MYSI


# 1.1.3 Combining MYSI datasets SSP1-5

# In[27]:


# 4&5

education45 = pd.merge(SSP4MYSI,SSP5MYSI,on=['Year','ISOCode'])
education45

#123&45
education=pd.merge(education321,education45,how='outer',on=['Year','ISOCode'])

#fill the blanks in SSP4&5
education["SSP4Years"].fillna(education["SSP1Years"],inplace=True)
education["SSP4MYSI"].fillna(education["SSP1MYSI"],inplace=True)
education["SSP5Years"].fillna(education["SSP1Years"],inplace=True)
education["SSP5MYSI"].fillna(education["SSP1MYSI"],inplace=True)


# delete NaN
education=education.dropna(subset=['Area'])

education = education.reset_index(drop=True)

# drop the country/region with incomplete data
education1 = education.drop(index = education[(education.ISOCode == 52)].index.tolist())
education1 = education1.drop(index = education1[(education1.ISOCode == 96)].index.tolist())
education1 = education1.drop(index = education1[(education1.ISOCode == 262)].index.tolist())
education1 = education1.drop(index = education1[(education1.ISOCode == 232)].index.tolist())
education1 = education1.drop(index = education1[(education1.ISOCode == 308)].index.tolist())
education1 = education1.drop(index = education1[(education1.ISOCode == 434)].index.tolist())
education1 = education1.drop(index = education1[(education1.ISOCode == 478)].index.tolist())
education1 = education1.drop(index = education1[(education1.ISOCode == 598)].index.tolist())
education1 = education1.drop(index = education1[(education1.ISOCode == 860)].index.tolist())

education1 = education1.reset_index(drop=True)

education1


# 1.2 Health Index-Life Expectancy of Birth 

# 1.2.1 Life Index SSP1-3

# In[29]:


#population

#123&45
#education123 
population123 = pd.read_csv(r'D:\HDI-Python\WCDE Data\population size (SSP1-3).csv')
population123

#when you only need one part
#population123 = population123['Scenario'].str.replace('SSP','0')
population123.replace({'SSP1':1,'SSP2':2,'SSP3':3,'Both':0,'Male':1,'Female':2},inplace=True)

population123


# In[30]:


# Life expectancy at Birth (SSP1-3)

leb123 = pd.read_csv(r'D:\HDI-Python\WCDE Data\LEB123.csv')
leb123

leb123['Year']=leb123['Period'].str[5:9]
leb123.replace({'SSP1':1,'SSP2':2,'SSP3':3,'Male':1,'Female':2},inplace=True)
leb123

#population & life expectancy
population123['Year']=population123['Year'].astype(int)
population123['Scenario']=population123['Scenario'].astype(int)
population123['ISOCode']=population123['ISOCode'].astype(int)
population123['Sex']=population123['Sex'].astype(int)

leb123['Year']=leb123['Year'].astype(int)
leb123['Scenario']=leb123['Scenario'].astype(int)
leb123['ISOCode']=leb123['ISOCode'].astype(int)
leb123['Sex']=leb123['Sex'].astype(int)


life123 = pd.merge(population123,leb123,how = 'outer',on=['Scenario','Area','Year','ISOCode','Sex'])

#delete period
life123 = life123.drop(columns=['Period'])
#SSP4mys_wide = SSP4mys_wide.reset_index(drop=True)

#delete Year=1950 since it's absent in LEB
life123 = life123.drop(life123[(life123['Year']<1951)].index)
life123 = life123.reset_index(drop=True)

life123


# In[31]:


#transfer the table to: both, male ,female

# sex= both 0
both = life123.drop(life123[(life123['Sex']>0)].index)

both = both.drop(columns=['Years'])
both.rename(columns={'Population':'Bothpop'},inplace = True)
both = both.reset_index(drop=True)

both


# In[32]:


#sex = male 1

#delete the row with value >1 or <1
male = life123.drop(life123[(life123['Sex']>1)|(life123['Sex']<1)].index)
male.rename(columns={'Population':'malepop','Years':'maleyear'},inplace = True)
male = male.reset_index(drop=True)

male


# In[33]:


#sex = female 2

#delete the row with value >1 or <1
female = life123.drop(life123[(life123['Sex']<2)].index)
female.rename(columns={'Population':'femalepop','Years':'femaleyear'},inplace = True)
female = female.reset_index(drop=True)

female


# 1.2.2 Life Index SSP4-5

# In[34]:


# marge both, male and female
le = pd.merge(both,male,how = 'outer',on=['Scenario','Area','Year','ISOCode'])
le = pd.merge(le,female,how = 'outer',on=['Scenario','Area','Year','ISOCode'])

le

#life expectancy at birth == male*male portion+female*female portion
le.loc[:,"leb"] = le['malepop']*le['maleyear']/le['Bothpop']+le['femalepop']*le['femaleyear']/le['Bothpop']

#LI
le.loc[:,"LI"] = (le['leb']-20)/90

le


# In[35]:


#life expectancy index
LI = le.drop(list(le)[5:12],axis = 1)
LI = LI.drop(columns=['Sex_x'])
LI = LI.reset_index(drop=True)
LI

#SSP1
LI1  = LI.drop(LI[(LI['Scenario']>1)].index)
LI1.rename(columns={'leb':'SSP1leb','LI':'SSP1LI'},inplace = True)
LI1 = LI1.drop(columns=['Scenario'])
LI1 = LI1.reset_index(drop=True)

#SSP2
LI2  = LI.drop(LI[(LI['Scenario']<2)|(LI['Scenario']>2)].index)
LI2.rename(columns={'leb':'SSP2leb','LI':'SSP2LI'},inplace = True)
LI2 = LI2.drop(columns=['Scenario'])
LI2 = LI2.reset_index(drop=True)


#SSP3
LI3 = LI.drop(LI[(LI['Scenario']<3)].index)
LI3.rename(columns={'leb':'SSP3leb','LI':'SSP3LI'},inplace = True)
LI3 = LI3.drop(columns=['Scenario'])
LI3 = LI3.reset_index(drop=True)

LI#.describe()


# In[36]:


#merge 123
LI123 = pd.merge(LI1,LI2,how = 'outer',on=['Area','Year','ISOCode'])
LI123 = pd.merge(LI123,LI3,how = 'outer',on=['Area','Year','ISOCode'])

LI123#.describe()


# 1.2.2 Health Index SSP4&5

# In[40]:


#lifeexpectancy - SSP4
#use period data to represent point (eg.take 2010-2015 as 2015)

lifeexpectancy4 = pd.read_csv(r'D:\HDI-Python\WCDE Data\life expectancy4.csv')

lifeexpectancy4.loc[:,'Year']=lifeexpectancy4['year']+5
lifeexpectancy4 = lifeexpectancy4.drop(columns=['ageno','eduno','year'])
lifeexpectancy4.rename(columns={'e0':'SSP4leb','isono':'ISOCode'},inplace = True)

lifeexpectancy4


# In[42]:


lifeexpectancy5 = pd.read_csv(r'D:\HDI-Python\WCDE Data\life expectancy5.csv')

lifeexpectancy5.loc[:,'Year']=lifeexpectancy5['year']+5
lifeexpectancy5 = lifeexpectancy5.drop(columns=['ageno','eduno','year'])
lifeexpectancy5.rename(columns={'e0':'SSP5leb','isono':'ISOCode'},inplace = True)

lifeexpectancy5


# In[43]:


#merge 4&5
leb45 = pd.merge(lifeexpectancy4,lifeexpectancy5,how = 'outer',on=['ISOCode','Year','sexno'])
leb45

# sex = female 2
female45 = leb45.drop(leb45[(leb45['sexno']<2)].index)
female45.rename(columns={'SSP4leb':'SSP4lebfemale','SSP5leb':'SSP5lebfemale'},inplace = True)
female45 = female45.drop(columns=['sexno'])
female45 = female45.reset_index(drop=True)

female45

# sex = male 1
male45 = leb45.drop(leb45[(leb45['sexno']>1)].index)
male45.rename(columns={'SSP4leb':'SSP4lebmale','SSP5leb':'SSP5lebmale'},inplace = True)
male45 = male45.drop(columns=['sexno'])
male45 = male45.reset_index(drop=True)

male45

#merge45
lifeexpectancy45 = pd.merge(male45,female45,how = 'outer',on=['ISOCode','Year'])
lifeexpectancy45 = lifeexpectancy45[['ISOCode','Year','SSP4lebmale','SSP4lebfemale','SSP5lebmale','SSP5lebfemale']]

lifeexpectancy45


# In[44]:


#population data-SSP4

SSP4epop = pd.read_csv(r'D:\HDI-Python\WCDE Data\Global population and human capital projections\SSP4epop_wide.csv')

#choose eduno=0 , all people, ageno=0 
SSP4epop = SSP4epop.drop(SSP4epop[(SSP4epop['eduno']>0)].index)
SSP4epop = SSP4epop.drop(columns=SSP4epop.columns[5:26], axis = 1)
#delete eduno column
SSP4epop = SSP4epop.drop(columns=['eduno'])
SSP4epop = SSP4epop.reset_index(drop=True)
SSP4epop


# sex = both (sexno = 0); education = total (eduno = 0)； ageno>4
SSP4epopboth = SSP4epop.drop(SSP4epop[(SSP4epop['sexno']>0)].index)
SSP4epopboth.rename(columns={'ageno_0':'SSP4both'},inplace = True)
SSP4epopboth = SSP4epopboth.drop(columns=['sexno'])
SSP4epopboth = SSP4epopboth.reset_index(drop=True)

SSP4epopboth

# sex = female 2
SSP4epopmale = SSP4epop.drop(SSP4epop[(SSP4epop['sexno']>1)|(SSP4epop['sexno']<1)].index)
SSP4epopmale.rename(columns={'ageno_0':'SSP4male'},inplace = True)
SSP4epopmale = SSP4epopmale.drop(columns=['sexno'])
SSP4epopmale = SSP4epopmale.reset_index(drop=True)

SSP4epopmale

# sex = male 1
SSP4epopfemale = SSP4epop.drop(SSP4epop[(SSP4epop['sexno']<2)].index)
SSP4epopfemale.rename(columns={'ageno_0':'SSP4female'},inplace = True)
SSP4epopfemale = SSP4epopfemale.drop(columns=['sexno'])
SSP4epopfemale = SSP4epopfemale.reset_index(drop=True)

SSP4epopfemale


# In[45]:


#merge both , male ,female

population4 = pd.merge(SSP4epopboth,SSP4epopmale,how = 'outer',on=['year','isono'])
population4 = pd.merge(population4,SSP4epopfemale,how = 'outer',on=['year','isono'])

population4


# In[46]:


#population data-SSP5

SSP5epop = pd.read_csv(r'D:\HDI-Python\WCDE Data\Global population and human capital projections\SSP5epop_wide.csv')

#choose eduno=0 , all people, ageno=0 
SSP5epop = SSP5epop.drop(SSP5epop[(SSP5epop['eduno']>0)].index)
SSP5epop = SSP5epop.drop(columns=SSP5epop.columns[5:26], axis = 1)
#delete eduno column
SSP5epop = SSP5epop.drop(columns=['eduno'])
SSP5epop = SSP5epop.reset_index(drop=True)
SSP5epop


# sex = both (sexno = 0); education = total (eduno = 0)； ageno>5
SSP5epopboth = SSP5epop.drop(SSP5epop[(SSP5epop['sexno']>0)].index)
SSP5epopboth.rename(columns={'ageno_0':'SSP5both'},inplace = True)
SSP5epopboth = SSP5epopboth.drop(columns=['sexno'])
SSP5epopboth = SSP5epopboth.reset_index(drop=True)

SSP5epopboth

# sex = female 2
SSP5epopmale = SSP5epop.drop(SSP5epop[(SSP5epop['sexno']>1)|(SSP5epop['sexno']<1)].index)
SSP5epopmale.rename(columns={'ageno_0':'SSP5male'},inplace = True)
SSP5epopmale = SSP5epopmale.drop(columns=['sexno'])
SSP5epopmale = SSP5epopmale.reset_index(drop=True)

SSP5epopmale

# sex = male 1
SSP5epopfemale = SSP5epop.drop(SSP5epop[(SSP5epop['sexno']<2)].index)
SSP5epopfemale.rename(columns={'ageno_0':'SSP5female'},inplace = True)
SSP5epopfemale = SSP5epopfemale.drop(columns=['sexno'])
SSP5epopfemale = SSP5epopfemale.reset_index(drop=True)

SSP5epopfemale


# In[47]:


#merge both , male ,female

population5 = pd.merge(SSP5epopboth,SSP5epopmale,how = 'outer',on=['year','isono'])
population5 = pd.merge(population5,SSP5epopfemale,how = 'outer',on=['year','isono'])

population5


# In[48]:


# merge 4&5 -population
population45 = pd.merge(population4,population5,how = 'outer',on=['year','isono'])
population45.rename(columns={'isono':'ISOCode','year':'Year'},inplace = True)

population45


# In[49]:


# collect 123 population data

# use the scenario SSP1 
pop1 = le.drop(le[(le['Scenario']>1)].index)

pop1 = pop1.drop(columns=['Sex_x','Sex_y','Sex','maleyear','femaleyear','leb','LI','Scenario'])
pop1 = pop1.reset_index(drop=True)

pop1

pop2 = le.drop(le[(le['Scenario']<2)|(le['Scenario']>2)].index)

pop2 = pop2.drop(columns=['Sex_x','Sex_y','Sex','maleyear','femaleyear','leb','LI','Scenario'])
pop2.rename(columns={'Bothpop':'Bothpop2','malepop':'malepop2','femalepop':'femalepop2'},inplace = True)
pop2 = pop2.reset_index(drop=True)

pop2

pop3 = le.drop(le[(le['Scenario']<3)].index)

pop3 = pop3.drop(columns=['Sex_x','Sex_y','Sex','maleyear','femaleyear','leb','LI','Scenario'])
pop3.rename(columns={'Bothpop':'Bothpop3','malepop':'malepop3','femalepop':'femalepop3'},inplace = True)
pop3 = pop3.reset_index(drop=True)

pop3


# In[105]:


#merge 123&45

pop = pd.merge(pop1,population45,how = 'outer',on=['Year','ISOCode'])
#pop = pop.drop(columns=['Scenario'])

# fill the 123 history data to 45
pop["SSP4both"].fillna(pop["Bothpop"],inplace=True)
pop["SSP4male"].fillna(pop["malepop"],inplace=True)
pop["SSP4female"].fillna(pop["femalepop"],inplace=True)
pop["SSP5both"].fillna(pop["Bothpop"],inplace=True)
pop["SSP5male"].fillna(pop["malepop"],inplace=True)
pop["SSP5female"].fillna(pop["femalepop"],inplace=True)


pop


# In[51]:


# merge population
population = pd.merge(pop1,pop2,how = 'outer',on=['Area','Year','ISOCode'])
population = pd.merge(population,pop3,how = 'outer',on=['Area','Year','ISOCode'])
population.rename(columns={'Bothpop':'Bothpop1','malepop':'malepop1','femalepop':'femalepop1'},inplace = True)
population = pd.merge(population,pop,how = 'outer',on=['Area','Year','ISOCode'])

population = population.drop(columns=['Bothpop','malepop','femalepop'])

population.drop(population[np.isnan(population['Bothpop1'])].index, inplace=True)

population


# In[52]:


# calculation of SSP45 LI
LI45 = pd.merge(pop,lifeexpectancy45,how = 'outer',on=['ISOCode','Year'])

# delete the rows that contain NaN in life expectancy data
LI45.drop(LI45[np.isnan(LI45['SSP4lebmale'])].index, inplace=True)
LI45.drop(LI45[np.isnan(LI45['SSP4female'])].index, inplace=True)

# calculate the LE 
LI45.loc[:,"SSP4leb"] = LI45['SSP4male']*LI45['SSP4lebmale']/LI45['SSP4both']+LI45['SSP4female']*LI45['SSP4lebfemale']/LI45['SSP4both']
LI45.loc[:,"SSP5leb"] = LI45['SSP5male']*LI45['SSP5lebmale']/LI45['SSP5both']+LI45['SSP5female']*LI45['SSP5lebfemale']/LI45['SSP5both']

#LI
LI45.loc[:,"SSP4LI"] = (LI45['SSP4leb']-20)/90
LI45.loc[:,"SSP5LI"] = (LI45['SSP5leb']-20)/90

# calculate the LI

LI45#.describe()


# In[53]:


#merge 123 &4

lifeexpectancy = pd.merge(LI123,LI45,how = 'outer',on=['Area','ISOCode','Year'])

#drop 116-cambodia

lifeexpectancy


# 1.3 Income Index-GDP data

# In[55]:


gdppc_ssp_data = pd.read_excel(r'D:\HDI-Python\WCDE Data\gdppc_ssp_data.xlsx')
gdppc_ssp_data

# only keep the gdppc data
gdppc_ssp_data = gdppc_ssp_data.drop(list(gdppc_ssp_data)[7:17],axis=1)

gdppc_ssp_data.dtypes

#calculate the GNI income
gdppc_ssp_data.loc[:,"SSP1income"] = (np.log(gdppc_ssp_data['ssp1_gdppc'])-np.log(100))/(np.log(402000)-np.log(100))
gdppc_ssp_data.loc[:,"SSP2income"] = (np.log(gdppc_ssp_data['ssp2_gdppc'])-np.log(100))/(np.log(402000)-np.log(100))
gdppc_ssp_data.loc[:,"SSP3income"] = (np.log(gdppc_ssp_data['ssp3_gdppc'])-np.log(100))/(np.log(402000)-np.log(100))
gdppc_ssp_data.loc[:,"SSP4income"] = (np.log(gdppc_ssp_data['ssp4_gdppc'])-np.log(100))/(np.log(402000)-np.log(100))
gdppc_ssp_data.loc[:,"SSP5income"] = (np.log(gdppc_ssp_data['ssp5_gdppc'])-np.log(100))/(np.log(402000)-np.log(100))

gdppc_ssp_data 


# In[56]:


# fill the blanks from excel
incomefill = pd.read_excel(r'D:\HDI-Python\WCDE Data\incomefill.xlsx')
incomefill

#merge
incomeindex = pd.merge(gdppc_ssp_data,incomefill,how = 'outer',on=['iso3','yr'])

#fill
incomeindex["SSP1income"].fillna(incomeindex["SSP1incomefill"],inplace=True)
incomeindex["SSP2income"].fillna(incomeindex["SSP1incomefill"],inplace=True)
incomeindex["SSP3income"].fillna(incomeindex["SSP1incomefill"],inplace=True)
incomeindex["SSP4income"].fillna(incomeindex["SSP1incomefill"],inplace=True)
incomeindex["SSP5income"].fillna(incomeindex["SSP1incomefill"],inplace=True)

# change names
incomeindex.rename(columns={'iso3':'three','yr':'Year'},inplace = True)

#from incomefill calculate the gdpdata 
incomeindex.loc[:,"SSP1incomefillgdp"] = np.exp((incomeindex['SSP1incomefill'])*(np.log(402000)-np.log(100))+np.log(100))

#fill in the gdp data in SSP1-5
incomeindex["ssp1_gdppc"].fillna(incomeindex["SSP1incomefillgdp"],inplace=True)
incomeindex["ssp2_gdppc"].fillna(incomeindex["SSP1incomefillgdp"],inplace=True)
incomeindex["ssp3_gdppc"].fillna(incomeindex["SSP1incomefillgdp"],inplace=True)
incomeindex["ssp4_gdppc"].fillna(incomeindex["SSP1incomefillgdp"],inplace=True)
incomeindex["ssp5_gdppc"].fillna(incomeindex["SSP1incomefillgdp"],inplace=True)

incomeindex#.describe()


# In[57]:


# import ISOCode
isocode = pd.read_excel(r'D:\HDI-Python\WCDE Data\isocode.xlsx')
isocode


# In[58]:


isocode.rename(columns={'ISO_code3':'three','ISO-code2':'two'},inplace = True)
incomeindex1 = pd.merge(incomeindex,isocode,how = 'outer',on=['three'])

# delete the columns 
incomeindex1 = incomeindex1.drop(columns=['eng full','two'])
incomeindex1.drop(incomeindex1[np.isnan(incomeindex1['Year'])].index, inplace=True)

incomeindex1.rename(columns={'num':'ISOCode'},inplace = True)

incomeindex1


# 1.4 HDI calculation

# In[59]:


# merge three indexes

#1. education 

educationindex = education1.drop(columns=['SSP1Years','SSP2Years','SSP3Years','SSP4Years','SSP5Years'])

educationindex


#31 dates, 202 countries


# In[62]:


# 2.life expectancy index

healthindex = lifeexpectancy.drop(columns=['SSP1leb','SSP2leb','SSP3leb','SSP4leb','SSP5leb'])
healthindex = healthindex.drop(list(healthindex)[6:19],axis = 1)
healthindex.drop(healthindex[np.isnan(healthindex['SSP1LI'])].index, inplace=True)

healthindex


# In[63]:


#3.GDP index 

gdpindex = incomeindex1.drop(list(incomeindex1)[2:7],axis = 1)
gdpindex = gdpindex.drop(columns=['three','SSP1incomefill'])

gdpindex.describe()

gdpindex


# In[64]:


#population
population.pivot_table(index=[u'Area',u'ISOCode'],columns = [u'Year']).describe()

population
#[population.ISOCode==729] #.drop(population[(population['ISOCode']<52)|(population['ISOCode']>52)].index)

#pd.options.display.max_rows = None
#pd.options.display.max_columns = None
pd.pivot_table(population, index=[u'ISOCode',u'Area'], columns=['Year'])#.sum()


# In[65]:


#population portion (date =2010)
#population of the countries include/world population

6692439/(13920007-6972709.2)


# In[66]:


# calculating HDI 

# merge three index

allindex = pd.merge(educationindex,healthindex,how = 'outer',on=['Area','Year','ISOCode'])
allindex = pd.merge(allindex,gdpindex,how = 'outer',on=['Year','ISOCode'])

allindex  #including the NaN values

allindex1 = allindex.dropna(how = 'any')
allindex1 = allindex1.drop(columns=['eng abb'])

allindex1.reset_index(inplace=True)
allindex1 = allindex1.drop(columns=['index'])

allindex1


# In[77]:


# economy portion 
#GDP of the countries include/world total GDP

economyportion = pd.merge(allindex1,incomeindex1, on=['Year','ISOCode'])
economyportion

pd.pivot_table(economyportion, index = [u'Area',u'ISOCode'], columns = [u'Year'],values=['SSP1incomefillgdp_y']).describe()


# In[86]:


#deleting 116 Cambodia(life expectancy in 1980 is under 20)

#allindex1["ISOCode"].value_counts()

allindex1[(allindex1.ISOCode ==116)].index.tolist()

allindex1 = allindex1.drop(index = allindex1[(allindex1.ISOCode ==116)].index.tolist())

# deleting Hong Kong as a region
allindex1 = allindex1.drop(index = allindex1[(allindex1.ISOCode ==344)].index.tolist())

allindex1 = allindex1.drop(columns=['SSP1incomefillgdp']).reset_index(drop = True)

allindex1


# In[87]:


#calculating the HDI 

allindex1.loc[:,"SSP1HDI"]=stats.gmean(allindex1[['SSP1MYSI','SSP1LI','SSP1income']],axis=1)
allindex1.loc[:,"SSP2HDI"]=stats.gmean(allindex1[['SSP2MYSI','SSP2LI','SSP2income']],axis=1)
allindex1.loc[:,"SSP3HDI"]=stats.gmean(allindex1[['SSP3MYSI','SSP3LI','SSP3income']],axis=1)
allindex1.loc[:,"SSP4HDI"]=stats.gmean(allindex1[['SSP4MYSI','SSP4LI','SSP4income']],axis=1)
allindex1.loc[:,"SSP5HDI"]=stats.gmean(allindex1[['SSP5MYSI','SSP5LI','SSP5income']],axis=1)


allindex1


# In[88]:


#allindex2 for 2100
allindex2 = allindex1.drop(allindex1[(allindex1['Year']<2100)].index)

pd.pivot_table(allindex2,index = [u'Area',u'ISOCode'])

allindex2


# In[89]:


#merge pop data
allindex3 = pd.merge(allindex1,population,how = 'outer',on=['Area','Year','ISOCode'])
allindex3 = allindex3.dropna(how = 'any')
allindex3 = allindex3.drop(columns=['malepop1','femalepop1','malepop2','femalepop2','malepop3','femalepop3','SSP4male','SSP4female','SSP5male', 'SSP5female'])
allindex3.rename(columns={'SSP4both':'Bothpop4','SSP5both':'Bothpop5'},inplace = True)
#pd.options.display.max_rows = 80

allindex3


# In[90]:


# world HDI

hdipop = pd.merge(allindex1,population,how = 'outer',on=['Area','Year','ISOCode'])
hdipop = hdipop.dropna(how = 'any')

worldhdi = hdipop.drop(list(hdipop)[3:18],axis = 1)
worldhdi = worldhdi.reset_index(drop=True)
worldhdi = worldhdi.drop(columns=['malepop1','femalepop1','malepop2','femalepop2','malepop3','femalepop3','SSP4male','SSP4female','SSP5male', 'SSP5female'])
worldhdi = worldhdi.reset_index(drop=True)

#hdipop
worldhdi.rename(columns={'SSP4both':'Bothpop4','SSP5both':'Bothpop5'},inplace = True)
worldhdi

#hdipop.to_excel(r'D:\HDI-Python\WCDE Data\hdipop.xlsx', index =False)
hdipop.describe()

pd.set_option('max_colwidth',100)

worldhdi


# In[92]:


#population-weighted HDI
worldhdi.loc[:,"SSP1world"] = worldhdi['Bothpop1']*worldhdi['SSP1HDI']
worldhdi.loc[:,"SSP2world"] = worldhdi['Bothpop2']*worldhdi['SSP2HDI']
worldhdi.loc[:,"SSP3world"] = worldhdi['Bothpop3']*worldhdi['SSP3HDI']
worldhdi.loc[:,"SSP4world"] = worldhdi['Bothpop4']*worldhdi['SSP4HDI']
worldhdi.loc[:,"SSP5world"] = worldhdi['Bothpop5']*worldhdi['SSP5HDI']

worldhdi1 = pd.pivot_table(worldhdi,index=[u'Area',u'ISOCode'],columns=[u'Year']) 

worldhdi1["SSP1world"] = worldhdi1["SSP1world"]/worldhdi1["Bothpop1"].sum()
worldhdi1["SSP2world"] = worldhdi1["SSP2world"]/worldhdi1["Bothpop2"].sum()
worldhdi1["SSP3world"] = worldhdi1["SSP3world"]/worldhdi1["Bothpop3"].sum()
worldhdi1["SSP4world"] = worldhdi1["SSP4world"]/worldhdi1["Bothpop4"].sum()
worldhdi1["SSP5world"] = worldhdi1["SSP5world"]/worldhdi1["Bothpop5"].sum()

worldhdi1#.describe()


# In[93]:


#global HDI
SSP1HDI = worldhdi1["SSP1world"].sum()
SSP1HDI = SSP1HDI.reset_index()
#SSP1.columns=['Year','SSP1HDI']
SSP1HDI.rename(columns={0:'SSP1HDI'},inplace = True)
SSP1HDI

SSP2HDI = worldhdi1["SSP2world"].sum()
SSP2HDI = SSP2HDI.reset_index()
SSP2HDI.rename(columns={0:'SSP2HDI'},inplace = True)
SSP2HDI

SSP3HDI = worldhdi1["SSP3world"].sum()
SSP3HDI = SSP3HDI.reset_index()
SSP3HDI.rename(columns={0:'SSP3HDI'},inplace = True)
SSP3HDI

SSP4HDI = worldhdi1["SSP4world"].sum()
SSP4HDI = SSP4HDI.reset_index()
SSP4HDI.rename(columns={0:'SSP4HDI'},inplace = True)
SSP4HDI

SSP5HDI = worldhdi1["SSP5world"].sum()
SSP5HDI = SSP5HDI.reset_index()
SSP5HDI.rename(columns={0:'SSP5HDI'},inplace = True)
SSP5HDI


# In[106]:


#worldhdiresult = pd.DataFrame(SSP1,SSP2,SSP3,SSP4,SSP5)

worldhdiresult = pd.merge(SSP1HDI,SSP2HDI,on=['Year'])
worldhdiresult = pd.merge(worldhdiresult,SSP3HDI,on=['Year'])
worldhdiresult = pd.merge(worldhdiresult,SSP4HDI,on=['Year'])
worldhdiresult = pd.merge(worldhdiresult,SSP5HDI,on=['Year'])

worldhdiresult.reset_index()

#worldhdiresult.rename(columns = {'0_x':'SSP1HDI','0_y':'SSP2HDI','0_x':'SSP1HDI'},inplace = True)
worldhdiresult


# In[95]:


# world LI
hdipop

worldli = hdipop.drop(list(hdipop)[3:8],axis = 1)

worldli = worldli.drop(list(worldli)[8:18],axis = 1)
worldli = worldli.drop(columns=['malepop1','femalepop1','malepop2','femalepop2','malepop3','femalepop3','SSP4male','SSP4female','SSP5male', 'SSP5female'])
worldli.rename(columns={'SSP4both':'Bothpop4','SSP5both':'Bothpop5'},inplace = True)

worldli.loc[:,"SSP1world"] = worldli['Bothpop1']*worldli['SSP1LI']
worldli.loc[:,"SSP2world"] = worldli['Bothpop2']*worldli['SSP2LI']
worldli.loc[:,"SSP3world"] = worldli['Bothpop3']*worldli['SSP3LI']
worldli.loc[:,"SSP4world"] = worldli['Bothpop4']*worldli['SSP4LI']
worldli.loc[:,"SSP5world"] = worldli['Bothpop5']*worldli['SSP5LI']

worldli1 = pd.pivot_table(worldli,index=[u'Area',u'ISOCode'],columns=[u'Year']) 

worldli1["SSP1world"] = worldli1["SSP1world"]/worldli1["Bothpop1"].sum()
worldli1["SSP2world"] = worldli1["SSP2world"]/worldli1["Bothpop2"].sum()
worldli1["SSP3world"] = worldli1["SSP3world"]/worldli1["Bothpop3"].sum()
worldli1["SSP4world"] = worldli1["SSP4world"]/worldli1["Bothpop4"].sum()
worldli1["SSP5world"] = worldli1["SSP5world"]/worldli1["Bothpop5"].sum()

worldli1

worldli.describe()


# In[96]:



SSP1LI = worldli1["SSP1world"].sum()
SSP1LI = SSP1LI.reset_index()
#SSP1.columns=['Year','SSP1LI']
SSP1LI.rename(columns={0:'SSP1LI'},inplace = True)
SSP1LI

SSP2LI = worldli1["SSP2world"].sum()
SSP2LI = SSP2LI.reset_index()
SSP2LI.rename(columns={0:'SSP2LI'},inplace = True)
SSP2LI

SSP3LI = worldli1["SSP3world"].sum()
SSP3LI = SSP3LI.reset_index()
SSP3LI.rename(columns={0:'SSP3LI'},inplace = True)
SSP3LI

SSP4LI = worldli1["SSP4world"].sum()
SSP4LI = SSP4LI.reset_index()
SSP4LI.rename(columns={0:'SSP4LI'},inplace = True)
SSP4LI

SSP5LI = worldli1["SSP5world"].sum()
SSP5LI = SSP5LI.reset_index()
SSP5LI.rename(columns={0:'SSP5LI'},inplace = True)
SSP5LI


# In[97]:


worldliresult = pd.merge(SSP1LI,SSP2LI,on=['Year'])
worldliresult = pd.merge(worldliresult,SSP3LI,on=['Year'])
worldliresult = pd.merge(worldliresult,SSP4LI,on=['Year'])
worldliresult = pd.merge(worldliresult,SSP5LI,on=['Year'])

worldliresult.reset_index()

#worldliresult.rename(columns = {'0_x':'SSP1LI','0_y':'SSP2LI','0_x':'SSP1LI'},inplace = True)
worldliresult


# In[98]:


# world education
hdipop

worldei = hdipop.drop(list(hdipop)[8:18],axis = 1)
worldei = worldei.drop(list(worldei)[8:13],axis = 1)

worldei = worldei.drop(columns=['malepop1','femalepop1','malepop2','femalepop2','malepop3','femalepop3','SSP4male','SSP4female','SSP5male', 'SSP5female'])
worldei.rename(columns={'SSP4both':'Bothpop4','SSP5both':'Bothpop5'},inplace = True)

worldei.loc[:,"SSP1world"] = worldei['Bothpop1']*worldei['SSP1MYSI']
worldei.loc[:,"SSP2world"] = worldei['Bothpop2']*worldei['SSP2MYSI']
worldei.loc[:,"SSP3world"] = worldei['Bothpop3']*worldei['SSP3MYSI']
worldei.loc[:,"SSP4world"] = worldei['Bothpop4']*worldei['SSP4MYSI']
worldei.loc[:,"SSP5world"] = worldei['Bothpop5']*worldei['SSP5MYSI']

worldei1 = pd.pivot_table(worldei,index=[u'Area',u'ISOCode'],columns=[u'Year']) 

worldei1["SSP1world"] = worldei1["SSP1world"]/worldei1["Bothpop1"].sum()
worldei1["SSP2world"] = worldei1["SSP2world"]/worldei1["Bothpop2"].sum()
worldei1["SSP3world"] = worldei1["SSP3world"]/worldei1["Bothpop3"].sum()
worldei1["SSP4world"] = worldei1["SSP4world"]/worldei1["Bothpop4"].sum()
worldei1["SSP5world"] = worldei1["SSP5world"]/worldei1["Bothpop5"].sum()

worldei1


# In[99]:


SSP1EI = worldei1["SSP1world"].sum()
SSP1EI = SSP1EI.reset_index()
#SSP1.columns=['Year','SSP1EI']
SSP1EI.rename(columns={0:'SSP1EI'},inplace = True)
SSP1EI

SSP2EI = worldei1["SSP2world"].sum()
SSP2EI = SSP2EI.reset_index()
SSP2EI.rename(columns={0:'SSP2EI'},inplace = True)
SSP2EI

SSP3EI = worldei1["SSP3world"].sum()
SSP3EI = SSP3EI.reset_index()
SSP3EI.rename(columns={0:'SSP3EI'},inplace = True)
SSP3EI

SSP4EI = worldei1["SSP4world"].sum()
SSP4EI = SSP4EI.reset_index()
SSP4EI.rename(columns={0:'SSP4EI'},inplace = True)
SSP4EI

SSP5EI = worldei1["SSP5world"].sum()
SSP5EI = SSP5EI.reset_index()
SSP5EI.rename(columns={0:'SSP5EI'},inplace = True)
SSP5EI


# In[100]:


worldeiresult = pd.merge(SSP1EI,SSP2EI,on=['Year'])
worldeiresult = pd.merge(worldeiresult,SSP3EI,on=['Year'])
worldeiresult = pd.merge(worldeiresult,SSP4EI,on=['Year'])
worldeiresult = pd.merge(worldeiresult,SSP5EI,on=['Year'])

worldeiresult.reset_index()

#worldeiresult.rename(columns = {'0_x':'SSP1EI','0_y':'SSP2EI','0_x':'SSP1EI'},inplace = True)
worldeiresult1=worldeiresult.T#pivot_table(columns = ['Year'])
worldeiresult1.rename(index={'SSP1EI':'SSP1','SSP2EI':'SSP2','SSP3EI':'SSP3','SSP4EI':'SSP4','SSP5EI':'SSP5'},inplace = True)
worldeiresult1


# In[107]:


# world income index
hdipop

worldii = hdipop.drop(list(hdipop)[3:13],axis = 1)
worldii = worldii.drop(list(worldii)[8:13],axis = 1)
worldii = worldii.drop(columns=['malepop1','femalepop1','malepop2','femalepop2','malepop3','femalepop3','SSP4male','SSP4female','SSP5male', 'SSP5female'])
worldii.rename(columns={'SSP4both':'Bothpop4','SSP5both':'Bothpop5'},inplace = True)

worldii.loc[:,"SSP1world"] = worldii['Bothpop1']*worldii['SSP1income']
worldii.loc[:,"SSP2world"] = worldii['Bothpop2']*worldii['SSP2income']
worldii.loc[:,"SSP3world"] = worldii['Bothpop3']*worldii['SSP3income']
worldii.loc[:,"SSP4world"] = worldii['Bothpop4']*worldii['SSP4income']
worldii.loc[:,"SSP5world"] = worldii['Bothpop5']*worldii['SSP5income']

worldii1 = pd.pivot_table(worldii,index=[u'Area',u'ISOCode'],columns=[u'Year']) 

worldii1["SSP1world"] = worldii1["SSP1world"]/worldii1["Bothpop1"].sum()
worldii1["SSP2world"] = worldii1["SSP2world"]/worldii1["Bothpop2"].sum()
worldii1["SSP3world"] = worldii1["SSP3world"]/worldii1["Bothpop3"].sum()
worldii1["SSP4world"] = worldii1["SSP4world"]/worldii1["Bothpop4"].sum()
worldii1["SSP5world"] = worldii1["SSP5world"]/worldii1["Bothpop5"].sum()

worldii1


# In[108]:


SSP1II = worldii1["SSP1world"].sum()
SSP1II = SSP1II.reset_index()
#SSP1.columns=['Year','SSP1II']
SSP1II.rename(columns={0:'SSP1II'},inplace = True)
SSP1II

SSP2II = worldii1["SSP2world"].sum()
SSP2II = SSP2II.reset_index()
SSP2II.rename(columns={0:'SSP2II'},inplace = True)
SSP2II

SSP3II = worldii1["SSP3world"].sum()
SSP3II = SSP3II.reset_index()
SSP3II.rename(columns={0:'SSP3II'},inplace = True)
SSP3II

SSP4II = worldii1["SSP4world"].sum()
SSP4II = SSP4II.reset_index()
SSP4II.rename(columns={0:'SSP4II'},inplace = True)
SSP4II

SSP5II = worldii1["SSP5world"].sum()
SSP5II = SSP5II.reset_index()
SSP5II.rename(columns={0:'SSP5II'},inplace = True)
SSP5II


# In[109]:


worldiiresult = pd.merge(SSP1II,SSP2II,on=['Year'])
worldiiresult = pd.merge(worldiiresult,SSP3II,on=['Year'])
worldiiresult = pd.merge(worldiiresult,SSP4II,on=['Year'])
worldiiresult = pd.merge(worldiiresult,SSP5II,on=['Year'])

worldiiresult.reset_index()

#worldiiresult.rename(columns = {'0_x':'SSP1II','0_y':'SSP2II','0_x':'SSP1II'},inplace = True)
worldiiresult1=worldiiresult.T#pivot_table(columns = ['Year'])
worldiiresult1.rename(index={'SSP1II':'SSP1','SSP2II':'SSP2','SSP3II':'SSP3','SSP4II':'SSP4','SSP5II':'SSP5'},inplace = True)
worldiiresult1


# In[110]:


#world scale three indexes

world = pd.merge(worldliresult,worldiiresult,on=['Year'])
world = pd.merge(world,worldeiresult,on=['Year'])
world = pd.merge(world,worldhdiresult,on=['Year'])

world


# In[112]:


#add in historical data
worldhdiresult.loc[:,'Historical Data']= worldhdiresult['SSP1HDI']
worldhdiresult.loc[9:26,'Historical Data'] = np.nan
worldhdiresult


# In[114]:


#coloring for different SSPs

#green = "2dfe54" #SSP1 bright light green
#yellow = "ffc512" #SSP2 sunflower
#blue = "02ccfe" #SSP3 bright sky blue
#red = "fe0002" #SSP4 fire engine red
#purple = "cb00f5" #SSP5 hot purple

#colors = [green,yellow,blue,red,purple]


# In[113]:


import matplotlib.pyplot as plt

plt.rcParams['font.family'].append(u'Arial')

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'

fig, ax= plt.subplots(figsize = (13,6))
fig.set_size_inches(13,6)

plt.plot( 'Year', 'SSP1HDI', data=worldhdiresult, marker='o', markerfacecolor='#2dfe54', markersize=5, color='#2dfe54', linewidth=2)
plt.plot( 'Year', 'SSP2HDI', data=worldhdiresult, marker='o', markerfacecolor='#ffc512', markersize=5, color='#ffc512', linewidth=2)
plt.plot( 'Year', 'SSP3HDI', data=worldhdiresult, marker='o', markerfacecolor='#02ccfe', markersize=5, color='#02ccfe', linewidth=2)
plt.plot( 'Year', 'SSP4HDI', data=worldhdiresult, marker='o', markerfacecolor='#fe0002', markersize=5, color='#fe0002', linewidth=2)
plt.plot( 'Year', 'SSP5HDI', data=worldhdiresult, marker='o', markerfacecolor='#cb00f5', markersize=5, color='#cb00f5', linewidth=2)
plt.plot( 'Year', 'Historical Data', data=worldhdiresult, marker='o', markerfacecolor='black', markersize=5, color='black', linewidth=2)

ax.set_xticks([year for year in np.arange(1970, 2110,10)])
ax.set_xticklabels(
    [year for year in np.arange(1970, 2110,10)] ,
    fontsize=22,
    weight=500
)

ax.tick_params(axis='x',rotation =45)

#plt.xlim((1970,2110))
plt.ylim((0.3,0.9))

font = {
    'family' :'Arial',
        'size': 22
        }

plt.xlabel('Year', fontdict = font)
plt.ylabel('HDI$_{Global}$', fontdict = font)

y_tick = np.arange(0.3,0.9,0.1)
plt.yticks(y_tick,fontsize = 22)


front_legend={
    'family' :'Arial',
    'weight':'normal',
    'size':18,
}

ax.legend(
      loc = 'lower right',
    frameon = False,
    facecolor = 'white',
    edgecolor = 'white',
    prop = front_legend
)

L=plt.legend(  loc = 'upper left',
    frameon = False,
    facecolor = 'white',
    edgecolor = 'white',
    prop = front_legend)
L.get_texts()[0].set_text('SSP1 Sustainability')
L.get_texts()[1].set_text('SSP2 Middle of the road')
L.get_texts()[2].set_text('SSP3 Regional rivalry')
L.get_texts()[3].set_text('SSP4 Inequality')
L.get_texts()[4].set_text('SSP5 Fossil-fueled development')
L.get_texts()[5].set_text('Historical Data')

fig=plt.gcf()

#plt.savefig('Global Population-weighted HDI.tif', dpi=1000, bbox_inches='tight')

plt.show()


# 1.2 Gini calculation

# In[146]:


#define function to calculate Gini coefficient

def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))


# In[152]:


ssp1hdigini = pd.Series(pd.pivot_table(worldhdi,index=[u'Area'],columns=[u'Year'],values=[u'SSP1HDI']).apply(gini),name='SSP1hdi')
ssp1hdigini
ssp2hdigini = pd.Series(pd.pivot_table(worldhdi,index=[u'Area'],columns=[u'Year'],values=[u'SSP2HDI']).apply(gini),name='SSP2hdi')
ssp2hdigini
ssp3hdigini = pd.Series(pd.pivot_table(worldhdi,index=[u'Area'],columns=[u'Year'],values=[u'SSP3HDI']).apply(gini),name='SSP3hdi')
ssp3hdigini
ssp4hdigini = pd.Series(pd.pivot_table(worldhdi,index=[u'Area'],columns=[u'Year'],values=[u'SSP4HDI']).apply(gini),name='SSP4hdi')
ssp4hdigini
ssp5hdigini = pd.Series(pd.pivot_table(worldhdi,index=[u'Area'],columns=[u'Year'],values=[u'SSP5HDI']).apply(gini),name='SSP5hdi')
ssp5hdigini

hdigini=pd.concat([ssp1hdigini,ssp2hdigini,ssp3hdigini,ssp4hdigini,ssp5hdigini],axis=1)
hdigini= pd.pivot_table(hdigini,index=[u'Year']).reset_index()
hdigini.loc[:,'Historical Data']= hdigini['SSP1hdi']
hdigini.loc[9:26,'Historical Data'] = np.nan

hdigini


# In[153]:


fig, ax= plt.subplots(figsize = (5,7))
fig.set_size_inches(5,7)

plt.rcParams['font.family'].append(u'Arial')

plt.plot( 'Year', 'SSP1hdi', data=hdigini, marker='o', markerfacecolor='#2dfe54', markersize=5, color='#2dfe54', linestyle='none')
plt.plot( 'Year', 'SSP2hdi', data=hdigini, marker='o', markerfacecolor='#ffc512', markersize=5, color='#ffc512', linestyle='none')
plt.plot( 'Year', 'SSP3hdi', data=hdigini, marker='o', markerfacecolor='#02ccfe', markersize=5, color='#02ccfe', linestyle='none')
plt.plot( 'Year', 'SSP4hdi', data=hdigini, marker='o', markerfacecolor='#fe0002', markersize=5, color='#fe0002', linestyle='none')
plt.plot( 'Year', 'SSP5hdi', data=hdigini, marker='o', markerfacecolor='#cb00f5', markersize=5, color='#cb00f5', linestyle='none')
plt.plot( 'Year', 'Historical Data', data=hdigini, marker='o', markerfacecolor='black', markersize=5, color='black', linestyle='none')

ax.set_xticks([year for year in np.arange(1970, 2100,40)])
ax.set_xticklabels(
    [year for year in np.arange(1970, 2100,40)] ,
    fontsize=22,
    weight=500
)

ax.tick_params(axis='x',rotation =45)

front_legend={
    'family':'Arial',
    'weight':'normal',
    'size':18,
}

ax.legend(
      loc = 'upper right',
    frameon = False,
    facecolor = 'white',
    edgecolor = 'white',
    prop = front_legend
    
)

L=plt.legend(  loc = 'upper right',
    frameon = False,
    facecolor = 'white',
    edgecolor = 'white',
    prop = front_legend)
L.get_texts()[0].set_text('SSP1')
L.get_texts()[1].set_text('SSP2')
L.get_texts()[2].set_text('SSP3')
L.get_texts()[3].set_text('SSP4')
L.get_texts()[4].set_text('SSP5')
L.get_texts()[5].set_text('Historical Data')

plt.ylim((0,0.25))

y_tick = np.arange(0,0.7,0.1)
plt.yticks(y_tick,fontsize = 22)

font = {
    'family' :'Arial',
        'size': 22
        }


plt.xlabel('Year', fontdict = font)
plt.ylabel('G$_{HDI}$', fontdict =font)

#plt.savefig('HDIgini1.tif', dpi=1000, bbox_inches='tight') 

plt.show()


# In[154]:


#health 
lifeexpectancy
healthgini = pd.merge(lifeexpectancy,worldhdi, on=['Year','ISOCode','Area']).dropna(axis = 0, how='any')
healthgini


# In[156]:


ssp1lebgini = pd.Series(pd.pivot_table(healthgini,index=[u'Area'],columns=[u'Year'],values=[u'SSP1leb']).apply(gini),name='SSP1leb')
ssp1lebgini
ssp2lebgini = pd.Series(pd.pivot_table(healthgini,index=[u'Area'],columns=[u'Year'],values=[u'SSP2leb']).apply(gini),name='SSP2leb')
ssp2lebgini
ssp3lebgini = pd.Series(pd.pivot_table(healthgini,index=[u'Area'],columns=[u'Year'],values=[u'SSP3leb']).apply(gini),name='SSP3leb')
ssp3lebgini
ssp4lebgini = pd.Series(pd.pivot_table(healthgini,index=[u'Area'],columns=[u'Year'],values=[u'SSP4leb']).apply(gini),name='SSP4leb')
ssp4lebgini
ssp5lebgini = pd.Series(pd.pivot_table(healthgini,index=[u'Area'],columns=[u'Year'],values=[u'SSP5leb']).apply(gini),name='SSP5leb')
ssp5lebgini

lebgini=pd.concat([ssp1lebgini,ssp2lebgini,ssp3lebgini,ssp4lebgini,ssp5lebgini],axis=1)
lebgini= pd.pivot_table(lebgini,index=[u'Year']).reset_index()
lebgini.loc[:,'Historical Data']= lebgini['SSP1leb']
lebgini.loc[9:26,'Historical Data'] = np.nan

lebgini


# In[157]:


fig, ax= plt.subplots(figsize = (5,7))
fig.set_size_inches(5,7)

plt.rcParams['font.family'].append(u'Arial')

plt.plot( 'Year', 'SSP1leb', data=lebgini,marker='o', markerfacecolor='#2dfe54', markersize=3, color='#2dfe54', linestyle='none')
plt.plot( 'Year', 'SSP2leb', data=lebgini, marker='o', markerfacecolor='#ffc512', markersize=3, color='#ffc512', linestyle='none')
plt.plot( 'Year', 'SSP3leb', data=lebgini, marker='o', markerfacecolor='#02ccfe', markersize=3, color='#02ccfe', linestyle='none')
plt.plot( 'Year', 'SSP4leb', data=lebgini, marker='o', markerfacecolor='#fe0002', markersize=3, color='#fe0002', linestyle='none')
plt.plot( 'Year', 'SSP5leb', data=lebgini, marker='o', markerfacecolor='#cb00f5', markersize=3, color='#cb00f5', linestyle='none')
plt.plot( 'Year', 'Historical Data', data=lebgini, marker='o', markerfacecolor='black', markersize=3, color='black', linestyle='none')

ax.set_xticks([year for year in np.arange(1970, 2100,40)])
ax.set_xticklabels(
    [year for year in np.arange(1970, 2100,40)] ,
    fontsize=22,
    weight=500
)

ax.tick_params(axis='x',rotation =45)

front_legend={
    'family':'Arial',
    'weight':'normal',
    'size':18,
}

y_tick = np.arange(0,0.7,0.1)
plt.yticks(y_tick,fontsize = 22)


plt.ylim((0,0.6))

font = {
    'family' :'Arial',
        'size': 22
        }


plt.xlabel('Year', fontdict = font)
plt.ylabel('G$_{Health}$', fontdict = font)

#plt.savefig('Ghealth.tif', dpi=1000, bbox_inches='tight') 

plt.show()


# In[158]:


#education 

education1
educationgini = pd.merge(education1,worldhdi, on=['Year','ISOCode','Area']).dropna(axis = 0, how='any')
educationgini#.describe()


# In[160]:


ssp1Yearsgini = pd.Series(pd.pivot_table(educationgini,index=[u'Area'],columns=[u'Year'],values=[u'SSP1Years']).apply(gini),name='SSP1Years')
ssp1Yearsgini
ssp2Yearsgini = pd.Series(pd.pivot_table(educationgini,index=[u'Area'],columns=[u'Year'],values=[u'SSP2Years']).apply(gini),name='SSP2Years')
ssp2Yearsgini
ssp3Yearsgini = pd.Series(pd.pivot_table(educationgini,index=[u'Area'],columns=[u'Year'],values=[u'SSP3Years']).apply(gini),name='SSP3Years')
ssp3Yearsgini
ssp4Yearsgini = pd.Series(pd.pivot_table(educationgini,index=[u'Area'],columns=[u'Year'],values=[u'SSP4Years']).apply(gini),name='SSP4Years')
ssp4Yearsgini
ssp5Yearsgini = pd.Series(pd.pivot_table(educationgini,index=[u'Area'],columns=[u'Year'],values=[u'SSP5Years']).apply(gini),name='SSP5Years')
ssp5Yearsgini

Yearsgini=pd.concat([ssp1Yearsgini,ssp2Yearsgini,ssp3Yearsgini,ssp4Yearsgini,ssp5Yearsgini],axis=1)
Yearsgini= pd.pivot_table(Yearsgini,index=[u'Year']).reset_index()
Yearsgini.loc[:,'Historical Data']= Yearsgini['SSP1Years']
Yearsgini.loc[9:26,'Historical Data'] = np.nan

Yearsgini


# In[162]:


fig, ax= plt.subplots(figsize = (5,7))
fig.set_size_inches(5,7)

plt.rcParams['font.family'].append(u'Arial')

plt.plot( 'Year', 'SSP1Years', data=Yearsgini, marker='o', markerfacecolor='#2dfe54', markersize=5, color='#2dfe54', linestyle='none')
plt.plot( 'Year', 'SSP2Years', data=Yearsgini, marker='o', markerfacecolor='#ffc512', markersize=5, color='#ffc512', linestyle='none')
plt.plot( 'Year', 'SSP3Years', data=Yearsgini, marker='o', markerfacecolor='#02ccfe', markersize=5, color='#02ccfe', linestyle='none')
plt.plot( 'Year', 'SSP4Years', data=Yearsgini, marker='o', markerfacecolor='#fe0002', markersize=5, color='#fe0002', linestyle='none')
plt.plot( 'Year', 'SSP5Years', data=Yearsgini, marker='o', markerfacecolor='#cb00f5', markersize=5, color='#cb00f5', linestyle='none')
plt.plot( 'Year', 'Historical Data', data=Yearsgini, marker='o', markerfacecolor='black', markersize=5, color='black', linestyle='none')

ax.set_xticks([year for year in np.arange(1970, 2100,40)])
ax.set_xticklabels(
    [year for year in np.arange(1970, 2100,40)] ,
    fontsize=22,
    weight=500
)


ax.tick_params(axis='x',rotation =45)

front_legend={
    'family':'Arial',
    'weight':'normal',
    'size':18,
}

y_tick = np.arange(0,0.7,0.1)
plt.yticks(y_tick,fontsize = 22)

plt.ylim((0,0.6))

font = {
    'family' :'Arial',
        'size': 22
        }

plt.xlabel('Year', fontdict = font)
plt.ylabel('G$_{Education}$', fontdict = font)

#plt.savefig('Geducation.tif', dpi=1000, bbox_inches='tight') 

plt.show()


# In[164]:


incomegini = pd.merge(incomeindex1,worldhdi, on=['Year','ISOCode']).dropna(axis = 0, how='any')
incomegini#.describe()


# In[165]:



ssp1gdpgini = pd.Series(pd.pivot_table(incomegini,index=[u'three'],columns=[u'Year'],values=[u'ssp1_gdppc']).apply(gini),name='ssp1gdpgini')
ssp1gdpgini
ssp2gdpgini = pd.Series(pd.pivot_table(incomegini,index=[u'three'],columns=[u'Year'],values=[u'ssp2_gdppc']).apply(gini),name='ssp2gdpgini')
ssp2gdpgini
ssp3gdpgini = pd.Series(pd.pivot_table(incomegini,index=[u'three'],columns=[u'Year'],values=[u'ssp3_gdppc']).apply(gini),name='ssp3gdpgini')
ssp3gdpgini
ssp4gdpgini = pd.Series(pd.pivot_table(incomegini,index=[u'three'],columns=[u'Year'],values=[u'ssp4_gdppc']).apply(gini),name='ssp4gdpgini')
ssp4gdpgini
ssp5gdpgini = pd.Series(pd.pivot_table(incomegini,index=[u'three'],columns=[u'Year'],values=[u'ssp5_gdppc']).apply(gini),name='ssp5gdpgini')
ssp5gdpgini

gdpgini=pd.concat([ssp1gdpgini,ssp2gdpgini,ssp3gdpgini,ssp4gdpgini,ssp5gdpgini],axis=1)
gdpgini= pd.pivot_table(gdpgini,index=[u'Year']).reset_index()
gdpgini.loc[:,'Historical Data']= gdpgini['ssp1gdpgini']
gdpgini.loc[9:26,'Historical Data'] = np.nan

gdpgini


# In[167]:


fig, ax= plt.subplots(figsize = (5,7))
fig.set_size_inches(5,7)

plt.rcParams['font.family'].append(u'Arial')

plt.plot( 'Year', 'ssp1gdpgini', data=gdpgini, marker='o', markerfacecolor='#2dfe54', markersize=5, color='#2dfe54', linestyle='none')
plt.plot( 'Year', 'ssp2gdpgini', data=gdpgini, marker='o', markerfacecolor='#ffc512', markersize=5, color='#ffc512', linestyle='none')
plt.plot( 'Year', 'ssp3gdpgini', data=gdpgini, marker='o', markerfacecolor='#02ccfe', markersize=5, color='#02ccfe', linestyle='none')
plt.plot( 'Year', 'ssp4gdpgini', data=gdpgini, marker='o', markerfacecolor='#fe0002', markersize=5, color='#fe0002', linestyle='none')
plt.plot( 'Year', 'ssp5gdpgini', data=gdpgini, marker='o', markerfacecolor='#cb00f5', markersize=5, color='#cb00f5', linestyle='none')
plt.plot( 'Year', 'Historical Data', data=gdpgini, marker='o', markerfacecolor='black', markersize=5, color='black', linestyle='none')

ax.set_xticks([year for year in np.arange(1970, 2100,40)])
ax.set_xticklabels(
    [year for year in np.arange(1970, 2100,40)] ,
    fontsize=22,
    weight=500
)

ax.tick_params(axis='x',rotation =45)

front_legend={
    'family':'Arial',
    'weight':'normal',
    'size':18,
}

y_tick = np.arange(0,0.7,0.1)
plt.yticks(y_tick,fontsize = 22)

ax.tick_params(axis='x',rotation =45)

plt.ylim((0,0.6))

font = {
    'family' :'Arial',
        'size': 22
        }


plt.xlabel('Year', fontdict = font)
plt.ylabel('G$_{Income}$', fontdict = font)

#plt.savefig('Gincome.tif', dpi=1000, bbox_inches='tight') 

plt.show()

