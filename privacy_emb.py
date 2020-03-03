
# coding: utf-8

# In[1]:


import sys
import pickle as pkl
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from input_data import load_data
from meansuring import get_score
from preprocessing import load_edges,preprocess_graph,sparse_to_tuple,construct_feed_dict
from process_attr import get_attr_list


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"




def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

        
del_all_flags(tf.flags.FLAGS)
flags = tf.app.flags
FLAGS = flags.FLAGS
# Settings
flags.DEFINE_string('f', '', 'Kernel')
flags.DEFINE_string('dataset', 'yale', 'Name of dateset')
flags.DEFINE_string('model', 'APPGE', 'Name of dateset')
 


# Load data
adj, features,adj_train, val_edges, val_edges_false, test_edges, test_edges_false,labels = load_data(FLAGS.dataset)


# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()


adj = adj_train

# Some preprocessing
adj_norm = preprocess_graph(adj)
features_mat = features.toarray()
attr_labels_list,dim_attr,features_rm_privacy = get_attr_list(FLAGS.dataset,labels,features_mat)

features_lil = sp.lil_matrix(features_rm_privacy)
features_tuple = sparse_to_tuple(features_lil .tocoo())
num_nodes = adj.shape[0]
features_sp = sparse_to_tuple(features_lil.tocoo())
num_features = features_sp[2][1]
features_nonzero = features_sp[1].shape[0]


pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = 1
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)



# In[2]:


#Change tuple to tensor
adj_tf = tf.SparseTensor(indices=adj_norm[0], values=adj_norm[1], dense_shape=adj_norm[2])
feats_tf = tf.SparseTensor(indices=features_tuple[0], values=features_tuple[1], dense_shape=features_tuple[2])


# In[3]:


###########Forward Propagation of GCN#################################

if FLAGS.model == 'APGE' or FLAGS.model == 'ADPGE':
    w1 = np.load('./data/'+FLAGS.model+'_'+FLAGS.dataset+'_w1.npy')
    w2 = np.load('./data/'+FLAGS.model+'_'+FLAGS.dataset+'_w2.npy')
    w3 = np.load('./data/'+FLAGS.model+'_'+FLAGS.dataset+'_w3.npy')
    b3 = np.load('./data/'+FLAGS.model+'_'+FLAGS.dataset+'_b3.npy')
    w1_tf = tf.constant(w1,dtype = tf.float64)
    w2_tf = tf.constant(w2,dtype = tf.float64)
    w3_tf = tf.constant(w3,dtype = tf.float64)
    b3_tf = tf.constant(b3,dtype = tf.float64)

    x = tf.sparse_tensor_dense_matmul(feats_tf, w1_tf)
    x = tf.sparse_tensor_dense_matmul(adj_tf, x)
    x = tf.nn.relu(x)
    x = tf.matmul(x, w2_tf)
    x_short = tf.sparse_tensor_dense_matmul(adj_tf, x)
    x= tf.layers.dense(inputs=x_short, units=64,activation=tf.nn.relu,kernel_initializer=tf.constant_initializer(w3),bias_initializer=tf.constant_initializer(b3))

else:
    w1 = np.load('./data/'+FLAGS.model+'_'+FLAGS.dataset+'_w1.npy')
    w2 = np.load('./data/'+FLAGS.model+'_'+FLAGS.dataset+'_w2.npy')
    w1_tf = tf.constant(w1,dtype = tf.float64)
    w2_tf = tf.constant(w2,dtype = tf.float64)
    x = tf.sparse_tensor_dense_matmul(feats_tf, w1_tf)
    x = tf.sparse_tensor_dense_matmul(adj_tf, x)
    x = tf.nn.relu(x)
    x = tf.matmul(x, w2_tf)
    x = tf.sparse_tensor_dense_matmul(adj_tf, x)
    


# In[4]:


emb = None
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    emb = x.eval()


# In[ ]:


###########Evaluate the embedding########################################################
link_acc,link_f1,p0_mlp,p1_mlp,p2_lr,p2_svm,p2_mlp,p0_f1,p1_f1,p2_lr_f1,p2_svm_f1,p2_mlp_f1,preds_all,labels_all,auc,ap = get_score(FLAGS.dataset,adj_orig,test_edges,test_edges_false,emb)
if FLAGS.dataset == 'rochestor':
    print('Uti Link ACC: ' + str(link_acc) +'\n')
    print('Uti Link F1: ' + str(link_f1)+'\n')
    print('Uti Attr MLP ACC: ' + str(p1_mlp)+'\n')
    print('Uti Attr MLP F1: ' + str(p1_f1)+'\n')
    print('Pri LR ACC: ' + str(p2_lr)+'\n')
    print('Pri LR F1: ' + str(p2_lr_f1)+'\n')
    print('Pri MLP ACC: ' + str(p2_mlp)+'\n')
    print('Pri MLP F1: ' + str(p2_mlp_f1)+'\n')
    print('Pri SVM ACC: ' + str(p2_svm)+'\n')
    print('Pri SVM F1: ' + str(p2_svm_f1)+'\n')
else:
    print('Uti Link ACC: ' + str(link_acc) +'\n')
    print('Uti Link F1: ' + str(link_f1)+'\n')
    print('Uti Attr MLP ACC: ' + str(p0_mlp)+'\n')
    print('Uti Attr MLP F1: ' + str(p0_f1)+'\n')
    print('Pri LR ACC: ' + str(p2_lr)+'\n')
    print('Pri LR F1: ' + str(p2_lr_f1)+'\n')
    print('Pri MLP ACC: ' + str(p2_mlp)+'\n')
    print('Pri MLP F1: ' + str(p2_mlp_f1)+'\n')
    print('Pri SVM ACC: ' + str(p2_svm)+'\n')
    print('Pri SVM F1: ' + str(p2_svm_f1)+'\n')

