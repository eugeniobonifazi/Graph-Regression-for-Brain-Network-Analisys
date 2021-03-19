import numpy as np
import pandas as pd
#from sklearn.tree.export import export_text



def open_file(file, limit, binary, absolute):
    
    mat = np.array(pd.read_csv(file, header = None, sep = ","))
    #mat=np.array(mat)
        
    if limit:
        #threshold = np.percentile(mat[np.tril_indices_from(abs(mat), k=-1)], limit)
        threshold = np.percentile(abs(mat), limit)
        mat[abs(mat)<threshold] = 0
 
    if binary:
        mat[abs(mat)>=threshold] = 1

    return mat



def open_file2(file):
    mat = np.array(pd.read_csv(file, header = None, sep = ","))

    threshold = np.percentile(mat[np.tril_indices_from(mat, k=-1)], 80)
    mat[mat<threshold] = 0
 
    mat[mat>=threshold] = 1

    return mat



def edges(file, sol1, sol2, pp):  
    net = open_file(file, pp)
    edges1 = np.sum(np.tril(np.array(subj)[np.ix_(sol1, sol1)], k = -1))
    edges2 = np.sum(np.tril(np.array(net)[np.ix_(sol2, sol2)], k = -1))
    return [edges1, edges2]



def l1_norm(file, sol, pp, avg_net1, avg_net2):  
    patient = np.tril(np.array(open_file(file, pp))[np.ix_(sol, sol)], k = -1)
    avg1 = np.tril(np.array(avg_net1)[np.ix_(sol, sol)], k = -1)
    avg2 = np.tril(np.array(avg_net2)[np.ix_(sol, sol)], k = -1)
    return[np.sum(abs(patient-avg1)), np.sum(abs(patient-avg2))]
 


def edges_1(file,dir1, sol1):  
    
    file1='{}/{}'.format(dir1, file)
    net = open_file2(file1)
    edges = np.sum(np.tril(np.array(net)[np.ix_(sol1, sol1)], k = -1))
    return (file[:-4], edges)




