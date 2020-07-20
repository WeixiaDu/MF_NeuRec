import argparse
import tensorflow as tf
import sys
import os.path
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from loadData import *

# from UNeuRec import UNeuRec
from INeuRec import INeuRec


def parse_args():
    parser = argparse.ArgumentParser(description='nnRec')
    parser.add_argument('--model', choices=['INeuRec', 'UNeuRec'], default='INeuRec')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_factors', type=int, default=16)
    parser.add_argument('--display_step', type=int, default=1500)
    parser.add_argument('--batch_size', type=int, default=3000)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--reg_rate', type=float, default=0.001)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    reg_rate = args.reg_rate
    num_factors = args.num_factors
    display_step = args.display_step
    batch_size = args.batch_size
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if args.model == "INeuRec":
        print("INeuRec")
        train_data, train_data_item, train_user_item_matrix, neg_train_matrix, test_data, test_matrix, num_users, num_items, num_training, unique_users \
            = load_movielens_ineurec(path="./data/ratings.dat", test_size=0.2,
                                     header=['user_id', 'item_id', 'rating','timestamp'], sep="\t")
        # print(unique_users)
        # print(len(unique_users))
        # print(type(unique_users))
        print('load data finished')

        with tf.Session(config=config) as sess:
            model = INeuRec(sess, num_users, num_items, num_training, num_factors, learning_rate, reg_rate, epochs,
                            batch_size, display_step)
            model.run(train_data_item, np.array(train_user_item_matrix), unique_users, neg_train_matrix, test_matrix,test_data)
    if args.model == "UNeuRec":
        print("UNeuRec")
        train_data, train_data_item, train_user_item_matrix, neg_train_matrix, test_data, test_matrix, num_users, num_items, num_training, unique_users \
            = load_movielens_uneurec(path="data/1m_ratings.dat", test_size=0.2, header=['user_id', 'item_id', 'rating', 'time'],
                                     sep="::")
        with tf.Session(config=config) as sess:
            model = UNeuRec(sess, num_users, num_items, num_training, num_factors, learning_rate, reg_rate, epochs,
                            batch_size, display_step)
            model.run(train_data_item, np.array(train_user_item_matrix), unique_users, neg_train_matrix, test_matrix)
