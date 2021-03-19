import pandas as pd
import argparse
import os
import numpy as np
from sklearn.cluster import KMeans


# Take Inputs

parser = argparse.ArgumentParser(description='Creating Dataset')


parser.add_argument('test', help='test type', type=str)
parser.add_argument('age', help='dataset upper age', type=int)
parser.add_argument('n_groups', help='dataset upper age', type=int) # number of groups 
parser.add_argument('criterion', help='test type', type=str) # E=equispaced, Q=quartiles, C=clustering
parser.add_argument('--a_c', help='split autism-control',
                    action='store_true')
args = parser.parse_args()



data = pd.read_csv("data/metadata.csv") #, usecols = ["SUB_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX", "DSM_IV_TR", "FILE_ID"])

#Exclude missing files
data = data[data["FILE_ID"] != "no_filename"]
groups = {1: "a", 2: "h"}
sex = {1: "m", 2: "f"}

data2=data[(data.AGE_AT_SCAN<=args.age)&(data[args.test]>0)&(data['{}_TEST_TYPE'.format(args.test)]=='WASI')]


# Equispaced
if args.criterion=='E':
    l=list(np.arange(data2['{}'.format(args.test)].min(),
                          data2['{}'.format(args.test)].max(),
                          (data2['{}'.format(args.test)].max()-data2['{}'.format(args.test)].min())/args.n_groups ))
 
    classes=[]
    intervals={}
    
    for i in range(len(l)):
        c='c{}'.format(i+1)
        classes.append(c)
        if l[i]!=l[-1]:
            intervals[c]=pd.Interval(l[i], l[i+1], closed='left')
        else:
            intervals[c]=pd.Interval(l[i], 1000, closed='left')

# Quartile
if args.criterion=='Q':
    l=[]
    for i in range(args.n_groups):
        l.append(list(data2[args.test].sort_values()[:int(round(len(data2)/args.n_groups))*i+1])[-1])
    classes=[]
    intervals={}
    
    for i in range(len(l)):
        c='c{}'.format(i+1)
        classes.append(c)
        if l[i]!=l[-1]:
            intervals[c]=pd.Interval(l[i], l[i+1],closed='left')
        else:
            intervals[c]=pd.Interval(l[i], 1000, closed='left')


# Clustering
if args.criterion=='C':
    kmeans = KMeans(n_clusters=args.n_groups, random_state=0).fit(np.asarray(data2['{}'.format(args.test)]).reshape(-1,1))



# Write file names for 
with open("files_names.txt", "w") as f:
    
    f.writelines([elem+"\n" for elem in data["FILE_ID"]])

    

for file in os.listdir("data/time_series"):
    
    file_pd = pd.read_csv("data/time_series/{}".format(file), skiprows = 1, sep = "\t", header = None)
    corr = file_pd.corr()
    nulls=file_pd.isnull().values.any()
    
    if corr.isna().sum().sum() != 0:
        continue
    if nulls:
        continue
    
    # group, if autistic or not
    group = groups[int(data["DX_GROUP"][data["FILE_ID"] == file[:-4]])]    
    
    data_file = data[data.FILE_ID==file[:-4]]
    
    age = data_file.AGE_AT_SCAN.item()<=args.age #and data_file.AGE_AT_SCAN.item()>10
    #test_type_name = "{}_TEST_TYPE".format(args.test) 
    #test_type = data_file[test_type_name].item()

    test_type_cond=data_file['FIQ_TEST_TYPE'].item()=='WASI' and data_file['VIQ_TEST_TYPE'].item()=='WASI'
    
    test_nonnull_cond = data_file.FIQ.item()>0 and data_file.VIQ.item()>0

    
    try:
        try_val = float(data[args.test][data['FILE_ID']== file[:-4]])
    except:
        continue
    
    if (age and test_type_cond and test_nonnull_cond and sex[data_file.SEX.item()]=='m'):
            
        iq = data_file[args.test].item()
        
        if args.criterion=='E' or args.criterion=='Q':
            med=[k for k,v in intervals.items() if iq in v][0]
        elif args.criterion=='C':
            med='c{}'.format(kmeans.predict(np.asarray(iq).reshape(-1,1))[0]+1)
        else:
            print('choose criterion')
                  
    else:
        continue
    
    pos_mat=np.nan_to_num(corr[corr>0], 0)
    neg_mat=np.nan_to_num(corr[corr<0], 0)
    np.fill_diagonal(pos_mat,0)
    if args.a_c:
        if not os.path.exists("datasets/under_{}/{}/pos/{}/criterion_{}/n_groups_{}/{}".format(args.age, group, args.test,args.criterion, args.n_groups,med)):
            os.makedirs("datasets/under_{}/{}/pos/{}/criterion_{}/n_groups_{}/{}".format(args.age,group, args.test, args.criterion, args.n_groups, med))
        np.savetxt(("datasets/under_{}/{}/pos/{}/criterion_{}/n_groups_{}/{}/{}.csv".format(args.age,group, args.test,args.criterion, args.n_groups, med, file[:-4])),pos_mat,delimiter=',',fmt='%.3f')
        if not os.path.exists("datasets/under_{}/{}/neg/{}/criterion_{}/n_groups_{}/{}".format(args.age, group, args.test,args.criterion, args.n_groups,med)):
            os.makedirs("datasets/under_{}/{}/neg/{}/criterion_{}/n_groups_{}/{}".format(args.age,group, args.test, args.criterion, args.n_groups, med))
        np.savetxt(("datasets/under_{}/{}/neg/{}/criterion_{}/n_groups_{}/{}/{}.csv".format(args.age,group, args.test,args.criterion, args.n_groups, med, file[:-4])),neg_mat,delimiter=',',fmt='%.3f')
        if not os.path.exists("datasets/under_{}/{}/original/{}/criterion_{}/n_groups_{}/{}".format(args.age, group, args.test,args.criterion, args.n_groups,med)):
            os.makedirs("datasets/under_{}/{}/original/{}/criterion_{}/n_groups_{}/{}".format(args.age,group, args.test, args.criterion, args.n_groups, med))
        np.savetxt(("datasets/under_{}/{}/original/{}/criterion_{}/n_groups_{}/{}/{}.csv".format(args.age,group, args.test,args.criterion, args.n_groups, med, file[:-4])),corr,delimiter=',',fmt='%.3f')
        
    else:
        if not os.path.exists("datasets/under_{}/{}/criterion_{}/n_groups_{}/{}".format(args.age,args.test,args.criterion, args.n_groups,med)):
            os.makedirs("datasets/under_{}/{}/criterion_{}/n_groups_{}/{}".format(args.age,args.test, args.criterion, args.n_groups, med))
        np.savetxt(("datasets/under_{}/{}/criterion_{}/n_groups_{}/{}/{}.csv".format(args.age,args.test,args.criterion, args.n_groups, med, file[:-4])),corr,delimiter=',',fmt='%.3f')
