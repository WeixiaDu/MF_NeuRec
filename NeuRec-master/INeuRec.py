import tensorflow as tf
import numpy as np
import time
import random
import math

from eval import *
import pickle


class INeuRec():
    def __init__(self, sess, num_users, num_items, num_training, num_factors, learning_rate, reg_rate, epochs,
                 batch_size, display_step):
        self.num_users = num_users
        self.num_items = num_items
        self.num_training = num_training
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.display_step = display_step
        self.reg_rate = reg_rate
        self.sess = sess
        # self.path_of_feature_file = './file/features/filetrust_Encoder_u500.pickle'
    def readItemFeature(self):
        with open(self.path_of_feature_file, 'rb') as fp:
            features = pickle.load(fp)
            return np.array(features)

    def run(self, train_data, train_user_item_matrix, unique_users, neg_train_matrix, test_matrix,test_data):

        # [512, 64]  [1024, 64]
        item_layers = [300,300,300,300,10]  
        user_layers = [150,150,150,150,10]
        layers=[500,256,128,10]
        self.cf_user_input = tf.placeholder(dtype=tf.int32, shape=[None], name='cf_user_input')
        self.cf_item_input = tf.placeholder(dtype=tf.int32, shape=[None], name='cf_item_input')
        self.y = tf.placeholder("float", [None], 'y')
        isTrain = tf.placeholder(tf.bool, shape=())
        self.keep_rate_net = tf.placeholder(tf.float32)
        self.keep_rate_input = tf.placeholder(tf.float32)
        self.keep_rate_output = tf.placeholder(tf.float32)

        # item_features = self.readItemFeature()
        # item_features = tf.constant(item_features,dtype=tf.float32)
        hidden_dim_6 = 10

        # R = tf.constant(train_user_item_matrix, dtype=tf.float32) #[3706,6040]  [1682,943]
        print('===========================',np.shape(train_user_item_matrix))
        R_u = tf.constant(train_user_item_matrix.T,dtype=tf.float32)
        # P = tf.Variable(tf.random.normal([self.num_users, hidden_dim_6], stddev=0.005))  #[6040,50]  [943,50]
        P_u = tf.Variable(tf.random.normal([self.num_items, hidden_dim_6], stddev=0.005))  #[6040,50]  [943,50]
        

        # user_id = tf.nn.embedding_lookup(P, self.cf_user_input) #[?,50]
        item_id = tf.nn.embedding_lookup(P_u,self.cf_item_input) #[?,50]
       
        user_factor = tf.cond(isTrain, lambda: tf.nn.dropout(tf.nn.embedding_lookup(R_u, self.cf_user_input), 0.97),
                              lambda: tf.nn.embedding_lookup(R_u, self.cf_user_input))      #[?,1682]

        # item_factor = tf.cond(isTrain, lambda: tf.nn.dropout(tf.nn.embedding_lookup(R, self.cf_item_input), 0.97),
        #                       lambda: tf.nn.embedding_lookup(R, self.cf_item_input))      #[?,1682]
        # print(np.shape(item_factor))

        for i in range(len(layers)):
            user_factor = tf.layers.dense(user_factor,layers[i],activation=tf.nn.relu)

        # for i in range(len(layers)):
        #     item_factor = tf.layers.dense(item_factor,layers[i],activation=tf.nn.relu)

        # self.pred_y = tf.reduce_sum(tf.nn.dropout(tf.multiply(user_id, item_factor), 1), 1)

        self.pred_y = tf.reduce_sum(tf.nn.dropout(tf.multiply(user_factor, item_id), 1), 1)  # tf.reshape(output, [-1])

        # self.pred_y = self.pred_y1 + self.pred_y2 
        # self.loss = - tf.reduce_sum(
        #     self.y * tf.log(self.pred_y) + (1 - self.y) * tf.log(1 - self.pred_y))
        self.loss = tf.reduce_sum(tf.square(self.y - self.pred_y)) 
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # initialize model
        init = tf.global_variables_initializer()
        total_batch = int(self.num_training / self.batch_size)
        print('total_batch:==========',total_batch)
        self.sess.run(init)

        temp = train_data.tocoo()

        item = temp.row.reshape(-1)
        user = temp.col.reshape(-1)
        rating = temp.data

        # train and test the model
        for epoch in range(self.epochs):
        # for epoch in range(1):
            idxs = np.random.permutation(self.num_training)
            # print(idxs)
            user_random = list(user[idxs])
            item_random = list(item[idxs])
            rating_random = list(rating[idxs])

            for i in range(total_batch):
            # for i in range(1):
                start_time = time.time()
                batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
                batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
                batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]

                _, c = self.sess.run([self.optimizer, self.loss], feed_dict={self.cf_user_input: batch_user,
                                                                             self.cf_item_input: batch_item,
                                                                             self.y: batch_rating,
                                                                             isTrain: True})
                avg_cost = c
                if i % self.display_step == 0:
                    print("Index: %04d; Epoch: %04d; cost= %.9f" % (i + 1, epoch, np.mean(avg_cost)))

            if (epoch) % (1) == 0 and epoch >= 0:

                pred_ratings_10 = {}
                pred_ratings_5 = {}
                pred_ratings = {}
                ranked_list = {}
                count = 0
                p_at_5 = []
                p_at_10 = []
                r_at_5 = []
                r_at_10 = []
                map1 = []
                mrr = []
                ndcg = []
                ndcg_at_5 = []
                ndcg_at_10 = []
                rmse_l = []
              
                learned_item_id = self.sess.run(item_id, feed_dict={self.cf_item_input: np.arange(self.num_items),
                                                                             isTrain: False})
                learned_user_factors = self.sess.run(user_factor,feed_dict={self.cf_user_input: np.arange(self.num_users),
                                                                        isTrain: False})

                # learned_user_id = self.sess.run(user_id, feed_dict={self.cf_user_input: np.arange(self.num_users),
                #                                                              isTrain: False})
                # learned_item_factors = self.sess.run(item_factor,feed_dict={self.cf_item_input: np.arange(self.num_items),
                #                                                         isTrain: False})
            
                # print('=========================')            
                # # print(np.shape(learned_user_factors))
                # f = open('./item_factors_'+str(hidden_dim_6)+'.pickle','wb')
                f = open('./user_factors'+str(hidden_dim_6)+'.pickle','wb')
                # pickle.dump(learned_item_factors,f)
                pickle.dump(learned_user_factors,f)
                f.close()

                # f = open('./item_factors'+str(hidden_dim_6)+'.pickle','wb')
                # f = open('./user_factors_'+str(hidden_dim_6)+'.pickle','wb')
                # 
                # f.close()
              
                # results = np.dot(learned_user_id, np.transpose(learned_item_factors))
                results = np.dot(learned_user_factors,np.transpose(learned_item_id))
                # res2 = np.dot(learned_user_id,np.transpose(learned_item_id))
                # results = res1 + res2
                print(np.shape(results))
                for u in unique_users:
                    # count += 1
                    user_neg_items = neg_train_matrix[u]
                    item_ids = []
                    scores = []
                    for j in user_neg_items:
                        item_ids.append(j)
                        scores.append(results[u, j])

                    neg_item_index = list(zip(item_ids, scores))

                    ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
                    pred_ratings[u] = [r[0] for r in ranked_list[u]]
                    pred_ratings_5[u] = pred_ratings[u][:5]
                    pred_ratings_10[u] = pred_ratings[u][:10]
                    p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, pred_ratings_5[u], test_matrix[u])
                    p_at_5.append(p_5)
                    r_at_5.append(r_5)
                    ndcg_at_5.append(ndcg_5)
                    p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, pred_ratings_10[u], test_matrix[u])
                    p_at_10.append(p_10)
                    r_at_10.append(r_10)
                    ndcg_at_10.append(ndcg_10)
                    map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], test_matrix[u])
                    map1.append(map_u)
                    mrr.append(mrr_u)
                    ndcg.append(ndcg_u)

                print("-------------------------------")
                f = open('res_100k.txt','a+')
                # print('rmse= ',np.mean(rmse_l))
                print(' Epoch: %04d;precision@5= %.4f, precision@10= %.4f, recall@5= %.4f, recall@10= %.4f, map= %.4f, mrr= %.4f' %(
                    epoch,np.mean(p_at_5),np.mean(p_at_10),np.mean(r_at_5),np.mean(r_at_10),np.mean(map1),np.mean(mrr)))

                f.write(' Epoch: %04d;precision@5= %.4f,precision@10= %.4f,recall@5= %.4f,recall@10= %.4f,map= %.4f,mrr= %.4f' %(
                    epoch,np.mean(p_at_5),np.mean(p_at_10),np.mean(r_at_5),np.mean(r_at_10),np.mean(map1),np.mean(mrr)))
                f.write('\n')
                f.close()
