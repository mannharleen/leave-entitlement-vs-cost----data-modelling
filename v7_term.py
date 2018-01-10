
# coding: utf-8
# pip, matplotlib, xlrd, pandas
# In[1]:

#Implements the for loop for iterations
import pandas as pd
import datetime
import sys
import math
#from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

print ':::INFO:::Number of arguments = %f' % (len(sys.argv[:])-1)
if (len(sys.argv[:]) <> 1 and len(sys.argv[:]) <> 10):
	sys.exit(':::ERROR::: Please input correct arguments. eg: python <scriptname> 1 2 3 4 5 6 7 8 9')

# In[2]:

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
#get_ipython().magic(u'matplotlib inline')
df = pd.read_excel('Annual Leave combined option 3 combined experimental.xlsm',sheetname=0,header=4)
##print df[['EMPLOYEE ID NO. ','YOS']][df['Entity'] == 'GMH']
df.columns=df.columns.map(lambda x: x.replace(' ','_'))
##sys.exit()
#df['num_years'] = df.DATE_JOINED.map(lambda x: (datetime.date.today().year - x.date().year))
df1 = df[['Entity', 'EMPLOYEE_ID_NO._', 'STANDARD', 'Grandfather?', 'FTE_replacement_needed?', 'Current', ]].copy()
df1['num_years'] = df.YOS.map(lambda x: math.floor(x))

df6 = pd.DataFrame(index=['PHSP','PHP','PHI','PHKL','PHC','PHA','PHK','PHAK','PHBP','PHM','GPG','GKL','GKK','GMH','PPP','PIR','MCO'],                          data=[82.1,147.666666666667,73.8666666666667,105.9,80.9,108.466666666667,68.6666666666667,82.7333333333333,73.9,88.1666666666667,103.966666666667,125.333333333333,126.466666666667,125.366666666667,109.5,115.3,383.133333333333                          ],columns=['avg_sal'])


# In[66]:

def func1(a,b,c,d,e,f,g,h,i):
                        
    #df2 contains end state values in pivot
    df2 = pd.DataFrame(columns=['Non-exec (S)','Non-exec (N)','Executive'], index=[0,1,2,3,4,5], \
    data=[[14+a, 16+b, 18+c],[14+a, 16+b, 18+c], [16+d, 18+e, 21+f], [16+d, 18+e, 21+f], [16+d, 18+e, 21+f], [18+g, 21+h, 24+i]])
    df3 = df2.stack().reset_index()
    df3.columns= ['num_years','STANDARD','Future']

    #make all num_years in d4 that are >5 = 5
    df4=df1.copy()
    df4.num_years=df4.num_years.map(lambda x: 5 if x>5 else x)

    df5 = pd.merge(df4,df3,how='left',left_on=['STANDARD','num_years'], right_on=['STANDARD','num_years'])
    df5['Diff'] = df5['Future']-df5['Current']
##    print df[['EMPLOYEE_ID_NO._','YOS']][df['Entity'] == 'GMH']
##    print df3
##    print df5[df5['Entity'] == 'GMH']
    #print df5.groupby('Entity').sum()
##    sys.exit()

    df7 = pd.merge(left=df5, right=df6, left_on='Entity', right_index=True)

    df7['buy_off_leave'] = df7.Diff * df7.avg_sal
    df7['buy_off_leave'] = df7['buy_off_leave'].map(lambda x: 0 if x >0 else x)
    df7.buy_off_leave[df7['Grandfather?'] == 'Y'] = 0

    df7['addn_FTE_cost'] = df7.Diff * df7.avg_sal
    df7['addn_FTE_cost'] = df7['addn_FTE_cost'].map(lambda x: 0 if x <0 else x)
    df7.addn_FTE_cost[df7['FTE_replacement_needed?'] == 'No'] = 0


    temp = df7.groupby('Entity').count()
    dop.total = temp['EMPLOYEE_ID_NO._']
    temp = df7[df7.Diff>0].groupby('Entity').count()
    dop.no_p_affected = temp.STANDARD
    temp = df7[(df7.Diff<0) & (df7['Grandfather?']=='N')].groupby('Entity').count() #[temp['Grandfather?']=='N']
    #print temp
    #sys.exit()
    dop.no_n_affected = temp.STANDARD
    temp = df7[df7.Diff>0].groupby('Entity').sum()
    dop.p_inc_AL = temp.Diff
    temp = df7[df7.Diff<0].groupby('Entity').sum()
    dop.n_inc_AL = temp.Diff
    dop.p_avg = dop.p_inc_AL / dop.no_p_affected
    dop.n_avg = abs(dop.n_inc_AL / dop.no_n_affected)
    temp = df7[df7['Grandfather?']=='Y'].groupby('Entity').count()
    dop.personal_to_holder = temp.Diff
    temp = df7.groupby('Entity').sum()
    dop.buy_off_leave = -1*temp.buy_off_leave
    temp = df7[df7['FTE_replacement_needed?'] == 'Yes'].groupby('Entity').sum()
    dop.addn_FTE_cost = temp.addn_FTE_cost



    pct_n_affected = dop.no_n_affected.sum() / dop.total.sum() * 100
    total_cost = dop.buy_off_leave.sum() + dop.addn_FTE_cost.sum()
    title = (a,b,c,d,e,f,g,h,i)
    dop_plt.shape[0]
    dop_plt.loc[dop_plt.shape[0]]=[title,pct_n_affected, total_cost]

    print('**************************************************************************************************************')
    print('***** Calculating for %s *****') % (str(a)+str(b)+str(c)+str(d)+str(e)+str(f)+str(g)+str(h)+str(i))
    #display(df2)
    print('***overall***')
    print dop
    #print('***summary***')
    #display(dop_plt)


# In[70]:

dop = pd.DataFrame(columns=['total','no_p_affected','p_inc_AL','p_avg', 'no_n_affected','n_inc_AL','n_avg','personal_to_holder','buy_off_leave','addn_FTE_cost'])
dop_plt = pd.DataFrame(columns=['title','pct_n_affected', 'total_cost'])

if (len(sys.argv[:]) == 10):
	func1(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7]),int(sys.argv[8]),int(sys.argv[9]))
else:
	for a in range(0,3):
		for b in range(0,3):
			for c in range(0,3):
				for d in range(0,3):
					for e in range(0,3):
						for f in range(0,3):
							for g in range(0,3):
								for h in range(0,3):
									for i in range(0,3):							
										func1(a,b,c,d,e,f,g,h,i)
##	sys.exit()


print('******************************************************************************************************************')
print('Iteration wise summary')
print(dop_plt)
dop_plt.to_csv('result.csv')

if (len(sys.argv[:]) == 1):
	p1= plt.figure() #(figsize=(15,10))
	
	#plt.plot(dop_plt.iloc[:,1],dop_plt.iloc[:,2])
	str(dop_plt.iloc[:,0])
	#color = np.random.rand(24)
	#plt.scatter(dop_plt.index,dop_plt.iloc[:,2],s=(dop_plt.iloc[:,1])**5/(10**5),c=color,alpha=0.5)
	plt.scatter(dop_plt.iloc[:,1],dop_plt.iloc[:,2])
	plt.xlabel('% -ve affected')
	plt.ylabel('Total cost')
	plt.title('Iteration VS Total cost')
	#plt.figtext(0.3,0.8,'size of bubbles represents iterations')
	plt.grid(b=True, which='major', color='b', linestyle='--')
	for i in range(0,len(dop_plt.iloc[:,1])):
		plt.annotate(i,(dop_plt.iloc[:,1][i],dop_plt.iloc[:,2][i]))
	#plt.show()
	
	#p2= plt.figure() #(figsize=(15,10))
	#dop_plt.pct_n_affected[(dop_plt.pct_n_affected <= 19) and (dop_plt.total_cost <= 1000000)]
	#print '$$$'
	#print dop_plt.pct_n_affected[(dop_plt.pct_n_affected<=25) & (dop_plt.total_cost<=1000000)].values
	#sys.exit()
	#plt.scatter(dop_plt.pct_n_affected[(dop_plt.pct_n_affected<=19) & (dop_plt.total_cost<=1000000)].values,dop_plt.total_cost[(dop_plt.pct_n_affected<=19) & (dop_plt.total_cost<=1000000)].values)
	#plt.xlabel('% -ve affected')
	#plt.ylabel('Total cost')
	#plt.title('Iteration VS Total cost')
	#plt.grid(b=True, which='major', color='b', linestyle='--')
	#for i in range(0,len(dop_plt.pct_n_affected[(dop_plt.pct_n_affected<=19) & (dop_plt.total_cost<=1000000)].values)):
		#plt.annotate(i,(dop_plt.iloc[:,1][i],dop_plt.iloc[:,2][i]))
		#plt.annotate(i,dop_plt.pct_n_affected[(dop_plt.pct_n_affected<=19) & (dop_plt.total_cost<=1000000)].values[i], \
		#dop_plt.total_cost[(dop_plt.pct_n_affected<=19) & (dop_plt.total_cost<=1000000)].values[i])
	plt.show()
	
        
