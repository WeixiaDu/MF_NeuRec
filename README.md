# MF_NeuRec
NeuRec模型训练
python INeuRec.py


在DeepRec-master/test/文件夹中运行python test_rating_pred.py
参数调节在DeepRec-master\models\rating_prediction\mf.py修改参数
self.item_path = './features/ml100k/feature/item_factors_60.pickle'
self.user_path = './features/ml100k/feature/user_factors60.pickle'
是通过NeuRec模型学习出的向量表示