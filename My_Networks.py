from functools import partial
import sys

from PyQt5.QtCore import QThread, pyqtSignal
from pandas import read_csv
import tensorflow as tf
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox, QInputDialog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LeakyReLU, Dense, Input, ReLU, ELU, Softmax, Concatenate
from tensorflow.keras.optimizers import Adam

import UI_rcwgan
import UI_regressor
from Shared_Api import check_dataset, Model_Structure
from os import path

# 全局变量供子线程使用
x_data = []
y_data = []
net = None
net_g = None
net_d = None
net_r = None
epoches = 0
criteon = None
optimizer = None


class MyThread(QThread):
    update_progress = pyqtSignal(int)
    update_train_loss = pyqtSignal(float)
    update_test_loss = pyqtSignal(float)
    training_end = pyqtSignal(bool)

    def __init__(self):
        super(MyThread, self).__init__()
        self.is_on = True

    def run(self):  # 线程执行函数
        x_train, x_test = train_test_split(x_data, test_size=0.3, random_state=1)
        y_train, y_test = train_test_split(y_data, test_size=0.3, random_state=1)

        # normalization
        x_train = MinMaxScaler().fit_transform(x_train)
        x_test = MinMaxScaler().fit_transform(x_test)
        y_train = MinMaxScaler().fit_transform(y_train.reshape(-1, 1))
        y_test = MinMaxScaler().fit_transform(y_test.reshape(-1, 1))

        db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
        db_val = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

        for epoch in range(epoches):
            # train
            for step, (x, y) in enumerate(db_train):
                if not self.is_on:
                    return
                with tf.GradientTape() as tape:

                    logits = net(x)  # [b, 1]
                    logits = tf.squeeze(logits, axis=1)  # [b]
                    # [b] vs [b]
                    train_loss = criteon(y, logits)
                grads = tape.gradient(train_loss, net.trainable_variables)
                optimizer.apply_gradients(zip(grads, net.trainable_variables))

            print('epoch:', epoch + 1, 'loss:', train_loss.numpy())
            # update UI
            self.update_progress.emit(epoch)
            if epoch > 19:
                self.update_train_loss.emit(train_loss)

            # test
            if epoch % 10 == 0:
                for x, y in db_val:
                    logits = net(x)  # [b, 1]
                    logits = tf.squeeze(logits, axis=1)  # [b]
                    # [b] vs [b]
                    test_loss = criteon(y, logits)

                print('epoch:', epoch + 1, 'val loss:', test_loss.numpy())
                # update UI
                if epoch > 19:
                    self.update_test_loss.emit(test_loss)

                x_train, x_test = train_test_split(x_data, test_size=0.3, random_state=1)
                y_train, y_test = train_test_split(y_data, test_size=0.3, random_state=1)

                # normalization
                x_train = MinMaxScaler().fit_transform(x_train)
                x_test = MinMaxScaler().fit_transform(x_test)
                y_train = MinMaxScaler().fit_transform(y_train.reshape(-1, 1))
                y_test = MinMaxScaler().fit_transform(y_test.reshape(-1, 1))

                db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
                db_val = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

        self.training_end.emit(True)


class rCWGAN(QWidget, UI_rcwgan.Ui_Form):
    def __init__(self, mWin):
        super(rCWGAN, self).__init__()
        global epoches, criteon, optimizer, x_data, y_data
        criteon = tf.keras.losses.MeanSquaredError()

        self.mWin = mWin

        # 如果放在__init__外，每次删除对象再实例化，内容仍是已删除对象的
        self.x_input_size = 17
        self.y_input_size = 1
        self.z_input_size = 10

        self.is_Loss_curves = True
        self.is_Curve_follow = False
        self.is_Load_Model_g = False
        self.is_Load_Model_d = False
        self.is_Load_Model_r = False
        self.dataset_name = None
        self.g_model_name = None
        self.d_model_name = None
        self.r_model_name = None

        self.output_act = LeakyReLU(alpha=0.2)
        self.train_loss = [0.0]
        self.test_loss = [0.0]
        self.predict = [0.0]

        # 引用一下
        self.x_data = x_data
        self.y_data = y_data

        self.setupUi(self)

        self.g_structure = Model_Structure()
        self.d_structure = Model_Structure()
        self.r_structure = Model_Structure()
        self.g_structure.treeWidget.topLevelItem(0).setText(0, '生成器')
        self.d_structure.treeWidget.topLevelItem(0).setText(0, '判别器')
        self.verticalLayout_2.addWidget(self.g_structure)
        self.verticalLayout_2.addWidget(self.d_structure)
        self.verticalLayout_2.addWidget(self.r_structure)

        # thread
        # self.my_thread = MyThread()
        # self.my_thread.update_progress.connect(self.UI_update)
        # self.my_thread.update_train_loss.connect(self.draw_trainLoss_update)
        # self.my_thread.update_test_loss.connect(self.draw_testLoss_update)
        # self.my_thread.training_end.connect(self.train_end)

        self.initUI()
        print('init model')

    def initUI(self):
        # slots and signal
        # self.bn_load_all.clicked.connect(partial(self.load_models_and_dataset, 'All'))
        self.bn_load_model_g.clicked.connect(partial(self.load_models_and_dataset, 'Generator'))
        self.bn_load_model_d.clicked.connect(partial(self.load_models_and_dataset, 'Discriminator'))
        self.bn_load_model_r.clicked.connect(partial(self.load_models_and_dataset, 'Regressor'))
        self.bn_load_dataset.clicked.connect(partial(self.load_models_and_dataset, 'Dataset'))
        self.bn_check_dataset.clicked.connect(self.check_dataset)
        # self.bn_train.clicked.connect(self.train_model)
        # self.bn_continue_train.clicked.connect(self.continue_train)
        self.g_structure.bn_saveStruct.clicked.connect(partial(self.save_model, 'Generator'))
        self.d_structure.bn_saveStruct.clicked.connect(partial(self.save_model, 'Discriminator'))
        self.r_structure.bn_saveStruct.clicked.connect(partial(self.save_model, 'Regressor'))
        # self.bn_saveWeight.clicked.connect(partial(self.save_model, 'weight'))
        # self.bn_predict.clicked.connect(self.make_predict)

    def check_dataset(self):
        fname = path.dirname(path.realpath(sys.argv[0]))
        fname, _ = QFileDialog.getOpenFileName(self, directory=fname + '\\' + 'data', caption='选择数据集文件',
                                               filter='数据集文件(*.csv)')
        if fname == '':
            return
        check_dataset(dataset=fname)

    def load_models_and_dataset(self, text):
        global x_data, y_data
        fname = path.dirname(path.realpath(sys.argv[0]))
        if text == 'All':
            dataset = str(fname + '\\' + 'data' + '\\' + 'PTA_DATA.csv')
            print('加载数据集:', dataset)
            data = read_csv(dataset, header=None)
            data = data.values

            x_data = data[:, :17]
            y_data = data[:, 17]
            # 引用一下
            self.x_data = x_data
            self.y_data = y_data

            print('train_data shape:', x_data.shape)
            print('train_label shape:', y_data.shape)

            self.dataset_name = dataset

            # self.model_name = fname + '\\' + 'model' + '\\' + 'r2.h5'
            # print('加载模型:', self.model_name)
            #
            # self.net = tf.keras.models.load_model(self.model_name)
            # self.is_Load_Model = True
            # self.bn_predict.setEnabled(True)
            # self.structure.treeWidget.topLevelItem(0).setText(0, '已加载的网络')
            # self.structure.bn_go_up.setEnabled(False)
            # self.structure.bn_go_down.setEnabled(False)
            # self.structure.bn_plus.setEnabled(False)
            # self.structure.bn_minus.setEnabled(False)
            # self.structure.bn_saveStruct.setEnabled(False)
            # self.bn_continue_train.setEnabled(True)
            # self.bn_train.setEnabled(False)
            # self.structure.treeWidget.topLevelItem(0).setExpanded(0)
            # self.structure.treeWidget.setEnabled(False)

            QMessageBox.information(self, '', '成功加载数据集和模型')

        elif text == 'Generator':
            fname, _ = QFileDialog.getOpenFileName(self, directory=fname + '\\' + 'model', caption='选择模型文件',
                                                   filter='模型文件(*.h5)')
            if fname == '':
                return
            print('加载模型:', fname)
            self.g_model_name = fname
            self.net_g = tf.keras.models.load_model(self.g_model_name)
            self.is_Load_Model_g = True
            self.g_structure.treeWidget.topLevelItem(0).setText(0, '已加载的网络')
            self.g_structure.bn_go_up.setEnabled(False)
            self.g_structure.bn_go_down.setEnabled(False)
            self.g_structure.bn_plus.setEnabled(False)
            self.g_structure.bn_minus.setEnabled(False)
            self.g_structure.bn_saveStruct.setEnabled(False)
            self.g_structure.treeWidget.topLevelItem(0).setExpanded(0)
            self.g_structure.treeWidget.setEnabled(False)
            QMessageBox.information(self, '', '成功加载模型')

        elif text == 'Discriminator':
            fname, _ = QFileDialog.getOpenFileName(self, directory=fname + '\\' + 'model', caption='选择模型文件',
                                                   filter='模型文件(*.h5)')
            if fname == '':
                return
            print('加载模型:', fname)
            self.d_model_name = fname
            self.net_d = tf.keras.models.load_model(self.d_model_name)
            self.is_Load_Model_d = True
            self.d_structure.treeWidget.topLevelItem(0).setText(0, '已加载的网络')
            self.d_structure.bn_go_up.setEnabled(False)
            self.d_structure.bn_go_down.setEnabled(False)
            self.d_structure.bn_plus.setEnabled(False)
            self.d_structure.bn_minus.setEnabled(False)
            self.d_structure.bn_saveStruct.setEnabled(False)
            self.d_structure.treeWidget.topLevelItem(0).setExpanded(0)
            self.d_structure.treeWidget.setEnabled(False)
            QMessageBox.information(self, '', '成功加载模型')

        elif text == 'Regressor':
            fname, _ = QFileDialog.getOpenFileName(self, directory=fname + '\\' + 'model', caption='选择模型文件',
                                                   filter='模型文件(*.h5)')
            if fname == '':
                return
            print('加载模型:', fname)
            self.r_model_name = fname
            self.net_r = tf.keras.models.load_model(self.r_model_name)
            self.is_Load_Model_r = True
            self.bn_predict.setEnabled(True)
            self.r_structure.treeWidget.topLevelItem(0).setText(0, '已加载的网络')
            self.r_structure.bn_go_up.setEnabled(False)
            self.r_structure.bn_go_down.setEnabled(False)
            self.r_structure.bn_plus.setEnabled(False)
            self.r_structure.bn_minus.setEnabled(False)
            self.r_structure.bn_saveStruct.setEnabled(False)
            self.r_structure.treeWidget.topLevelItem(0).setExpanded(0)
            self.r_structure.treeWidget.setEnabled(False)
            QMessageBox.information(self, '', '成功加载模型')

        elif text == 'Dataset':
            fname, _ = QFileDialog.getOpenFileName(self, directory=fname + '\\' + 'data', caption='选择数据集文件',
                                                   filter='数据集文件(*.csv)')
            if fname == '':
                return
            print('加载数据集:', fname)
            data = read_csv(fname, header=None)
            data = data.values

            x_data = data[:, :17]
            y_data = data[:, 17]
            # 引用一下
            self.x_data = x_data
            self.y_data = y_data

            print('train_data shape:', x_data.shape)
            print('train_label shape:', y_data.shape)

            self.dataset_name = fname
            QMessageBox.information(self, '', '成功加载数据集')

    def build_model_structure(self, text):
        if text == 'Generator':
            if not self.is_Load_Model_g:
                noise_label_layer = Input(shape=(self.y_input_size + self.z_input_size,), dtype=float, name="Generator_input_y_and_noise")

                self.net_g = tf.keras.models.Sequential()

                # add input_layer
                self.net_g.add(noise_label_layer)

                # add hidden_layers
                for i in range(len(self.g_structure.layers_list)):
                    activation = self.g_structure.layers_list[i].comboBox_2.currentText()
                    if activation == 'LeakyReLu':
                        k_initial = tf.keras.initializers.he_normal()
                    elif activation == 'relu':
                        k_initial = tf.keras.initializers.he_uniform()
                    elif activation == 'elu':
                        k_initial = tf.keras.initializers.he_normal()
                    else:
                        k_initial = tf.keras.initializers.RandomUniform()

                    # choose layer type
                    if self.g_structure.layers_list[i].lineEdit.text() == 'Dense':
                        self.net_g.add(Dense(units=self.g_structure.layers_list[i].spinBox.value(),
                                             kernel_initializer=k_initial,
                                             name='Dense_Layer_' + str(i)
                                             )
                                       )
                    elif self.g_structure.layers_list[i].lineEdit.text() == 'Conv':
                        pass

                    # add activation
                    if activation == 'LeakyReLu':
                        self.net_g.add(LeakyReLU(alpha=0.3))
                    elif activation == 'relu':
                        self.net_g.add(ReLU())
                    elif activation == 'elu':
                        self.net_g.add(ELU())
                    elif activation == 'softmax':
                        self.net_g.add(Softmax())

                self.net_g.add(Dense(units=self.x_input_size, kernel_initializer=tf.keras.initializers.RandomUniform(),
                                     name='Output_Layer'))

                print(self.net_g.summary())
        elif text == 'Discriminator':
            print('Discriminator')
        elif text == 'Regressor':
            if not self.is_Load_Model_r:
                self.net_r = tf.keras.models.Sequential()

                # add input_layer
                self.net_r.add(Input(shape=(self.x_input_size,),
                                     dtype=float,
                                     name='Input_Layer')
                               )

                # add hidden_layers
                for i in range(len(self.r_structure.layers_list)):
                    activation = self.r_structure.layers_list[i].comboBox_2.currentText()
                    if activation == 'LeakyReLu':
                        k_initial = tf.keras.initializers.he_normal()
                    elif activation == 'relu':
                        k_initial = tf.keras.initializers.he_uniform()
                    elif activation == 'elu':
                        k_initial = tf.keras.initializers.he_normal()
                    else:
                        k_initial = tf.keras.initializers.RandomUniform()

                    # choose layer type
                    if self.r_structure.layers_list[i].lineEdit.text() == 'Dense':
                        self.net_r.add(Dense(units=self.r_structure.layers_list[i].spinBox.value(),
                                             kernel_initializer=k_initial,
                                             name='Dense_Layer_' + str(i)
                                             )
                                       )
                    elif self.r_structure.layers_list[i].lineEdit.text() == 'Conv':
                        pass

                    # add activation
                    if activation == 'LeakyReLu':
                        self.net_r.add(LeakyReLU(alpha=0.3))
                    elif activation == 'relu':
                        self.net_r.add(ReLU())
                    elif activation == 'elu':
                        self.net_r.add(ELU())
                    elif activation == 'softmax':
                        self.net_r.add(Softmax())

                self.net_r.add(Dense(units=self.y_input_size, kernel_initializer=tf.keras.initializers.RandomUniform(),
                                     name='Output_Layer'))

                print(self.net_r.summary())

    def save_model(self, text):
        if text == 'Generator':
            txt, ok = QInputDialog.getText(self, '保存模型', '输入要保存模型的名字:')
            if txt == '':
                QMessageBox.warning(self, '', '输入为空')
            elif ok and txt:
                self.build_model_structure(text)
                self.g_model_name = path.dirname(path.realpath(sys.argv[0])) + '\\' + 'model' + '\\' + txt + '.h5'
                self.net_g.save(self.g_model_name)
                print('加载模型:', self.g_model_name)
                QMessageBox.information(self, '', '模型保存成功')
                self.g_structure.treeWidget.topLevelItem(0).setText(1, '当前模型->' + self.g_model_name)
        elif text == 'Discriminator':
            txt, ok = QInputDialog.getText(self, '保存模型', '输入要保存模型的名字:')
            if txt == '':
                QMessageBox.warning(self, '', '输入为空')
            elif ok and txt:
                self.build_model_structure(text)
                self.d_model_name = path.dirname(path.realpath(sys.argv[0])) + '\\' + 'model' + '\\' + txt + '.h5'
                self.net_d.save(self.d_model_name)
                print('加载模型:', self.d_model_name)
                QMessageBox.information(self, '', '模型保存成功')
                self.d_structure.treeWidget.topLevelItem(0).setText(1, '当前模型->' + self.d_model_name)
        elif text == 'Regressor':
            txt, ok = QInputDialog.getText(self, '保存模型', '输入要保存模型的名字:')
            if txt == '':
                QMessageBox.warning(self, '', '输入为空')
            elif ok and txt:
                self.build_model_structure(text)
                self.r_model_name = path.dirname(path.realpath(sys.argv[0])) + '\\' + 'model' + '\\' + txt + '.h5'
                self.net_r.save(self.r_model_name)
                print('加载模型:', self.r_model_name)
                QMessageBox.information(self, '', '模型保存成功')
                self.r_structure.treeWidget.topLevelItem(0).setText(1, '当前模型->' + self.r_model_name)
        else:
            if self.model_name == None:
                ok = QMessageBox.question(self, '', '需要先保存模型，是否保存当前的模型？', QMessageBox.Yes | QMessageBox.Cancel)
                if ok == QMessageBox.Yes:
                    self.save_model(text='struct')
                    self.save_model(text='weight')
            else:
                self.net.save(self.model_name)
                QMessageBox.information(self, '', '权重保存成功')


class Regressor(QWidget, UI_regressor.Ui_Form):
    def __init__(self, mWin):
        super(Regressor, self).__init__()
        global epoches, criteon, optimizer, x_data, y_data
        criteon = tf.keras.losses.MeanSquaredError()

        self.mWin = mWin

        # 如果放在__init__外，每次删除对象再实例化，内容仍是已删除对象的
        self.x_input_size = 17
        self.y_input_size = 1

        self.is_Loss_curves = True
        self.is_Curve_follow = False
        self.is_Load_Model = False
        self.dataset_name = None
        self.model_name = None

        self.output_act = LeakyReLU(alpha=0.2)
        self.train_loss = [0.0]
        self.test_loss = [0.0]
        self.predict = [0.0]

        # 引用一下
        self.x_data = x_data
        self.y_data = y_data

        self.setupUi(self)

        self.structure = Model_Structure()
        self.verticalLayout_2.addWidget(self.structure)

        # thread
        self.my_thread = MyThread()
        self.my_thread.update_progress.connect(self.UI_update)
        self.my_thread.update_train_loss.connect(self.draw_trainLoss_update)
        self.my_thread.update_test_loss.connect(self.draw_testLoss_update)
        self.my_thread.training_end.connect(self.train_end)

        self.initUI()
        print('init model')

    def initUI(self):
        # slots and signal
        self.bn_load_all.clicked.connect(partial(self.load_models_and_dataset, 'All'))
        self.bn_load_model.clicked.connect(partial(self.load_models_and_dataset, 'Regressor'))
        self.bn_load_dataset.clicked.connect(partial(self.load_models_and_dataset, 'Dataset'))
        self.bn_check_dataset.clicked.connect(self.check_dataset)
        self.bn_train.clicked.connect(self.train_model)
        self.bn_continue_train.clicked.connect(self.continue_train)
        self.structure.bn_saveStruct.clicked.connect(partial(self.save_model, 'struct'))
        self.bn_saveWeight.clicked.connect(partial(self.save_model, 'weight'))
        self.bn_predict.clicked.connect(self.make_predict)

    def check_dataset(self):
        fname = path.dirname(path.realpath(sys.argv[0]))
        fname, _ = QFileDialog.getOpenFileName(self, directory=fname + '\\' + 'data', caption='选择数据集文件',
                                               filter='数据集文件(*.csv)')
        if fname == '':
            return
        check_dataset(dataset=fname)

    def UI_update(self, epoch):
        self.progressBar.setValue(int((epoch + 1) / self.remainder))
        self.mWin.graph_widget.vline_axis = epoch
        if self.is_Curve_follow:
            if self.is_Loss_curves:
                if epoch > self.mWin.graph_widget.f_range:
                    self.mWin.graph_widget.train_graph.setXRange(epoch - self.mWin.graph_widget.f_range, epoch)
                else:
                    self.mWin.graph_widget.train_graph.setXRange(0, epoch)
                i = epoch / 10

                if i > self.mWin.graph_widget.f_range / 10:
                    self.mWin.graph_widget.test_graph.setXRange(i - self.mWin.graph_widget.f_range / 10, i)
                else:
                    self.mWin.graph_widget.test_graph.setXRange(0, epoch / 10)

    def draw_trainLoss_update(self, trainLoss):
        self.train_loss.append(trainLoss)
        # update UI
        if self.is_Loss_curves:
            self.mWin.graph_widget.curve.setData(self.train_loss)

    def draw_testLoss_update(self, testLoss):
        self.test_loss.append(testLoss)
        # update UI
        if self.is_Loss_curves:
            self.mWin.graph_widget.curve2.setData(self.test_loss)

    def load_models_and_dataset(self, text):
        global x_data, y_data
        fname = path.dirname(path.realpath(sys.argv[0]))
        if text == 'All':
            dataset = str(fname + '\\' + 'data' + '\\' + 'PTA_DATA.csv')
            print('加载数据集:', dataset)
            data = read_csv(dataset, header=None)
            data = data.values

            x_data = data[:, :17]
            y_data = data[:, 17]
            # 引用一下
            self.x_data = x_data
            self.y_data = y_data

            print('train_data shape:', x_data.shape)
            print('train_label shape:', y_data.shape)

            self.dataset_name = dataset

            self.model_name = fname + '\\' + 'model' + '\\' + 'r2.h5'
            print('加载模型:', self.model_name)

            self.net = tf.keras.models.load_model(self.model_name)
            self.is_Load_Model = True
            self.bn_predict.setEnabled(True)
            self.structure.treeWidget.topLevelItem(0).setText(0, '已加载的网络')
            self.structure.bn_go_up.setEnabled(False)
            self.structure.bn_go_down.setEnabled(False)
            self.structure.bn_plus.setEnabled(False)
            self.structure.bn_minus.setEnabled(False)
            self.structure.bn_saveStruct.setEnabled(False)
            self.bn_continue_train.setEnabled(True)
            self.bn_train.setEnabled(False)
            self.structure.treeWidget.topLevelItem(0).setExpanded(0)
            self.structure.treeWidget.setEnabled(False)

            QMessageBox.information(self, '', '成功加载数据集和模型')

        elif text == 'Regressor':
            fname, _ = QFileDialog.getOpenFileName(self, directory=fname + '\\' + 'model', caption='选择模型文件',
                                                   filter='模型文件(*.h5)')
            if fname == '':
                return
            print('加载模型:', fname)
            self.model_name = fname
            self.net = tf.keras.models.load_model(self.model_name)
            self.is_Load_Model = True
            self.bn_predict.setEnabled(True)
            self.structure.treeWidget.topLevelItem(0).setText(0, '已加载的网络')
            self.structure.bn_go_up.setEnabled(False)
            self.structure.bn_go_down.setEnabled(False)
            self.structure.bn_plus.setEnabled(False)
            self.structure.bn_minus.setEnabled(False)
            self.structure.bn_saveStruct.setEnabled(False)
            self.bn_continue_train.setEnabled(True)
            self.bn_train.setEnabled(False)
            self.structure.treeWidget.topLevelItem(0).setExpanded(0)
            self.structure.treeWidget.setEnabled(False)
            QMessageBox.information(self, '', '成功加载模型')

        elif text == 'Dataset':
            fname, _ = QFileDialog.getOpenFileName(self, directory=fname + '\\' + 'data', caption='选择数据集文件',
                                                   filter='数据集文件(*.csv)')
            if fname == '':
                return
            print('加载数据集:', fname)
            data = read_csv(fname, header=None)
            data = data.values

            x_data = data[:, :17]
            y_data = data[:, 17]
            # 引用一下
            self.x_data = x_data
            self.y_data = y_data

            print('train_data shape:', x_data.shape)
            print('train_label shape:', y_data.shape)

            self.dataset_name = fname
            QMessageBox.information(self, '', '成功加载数据集')

    def build_model_structure(self):
        if not self.is_Load_Model:
            self.net = tf.keras.models.Sequential()

            # add input_layer
            self.net.add(Input(shape=(self.x_input_size,),
                               dtype=float,
                               name='Input_Layer')
                         )

            # add hidden_layers
            for i in range(len(self.structure.layers_list)):
                activation = self.structure.layers_list[i].comboBox_2.currentText()
                if activation == 'LeakyReLu':
                    k_initial = tf.keras.initializers.he_normal()
                elif activation == 'relu':
                    k_initial = tf.keras.initializers.he_uniform()
                elif activation == 'elu':
                    k_initial = tf.keras.initializers.he_normal()
                else:
                    k_initial = tf.keras.initializers.RandomUniform()

                # choose layer type
                if self.structure.layers_list[i].lineEdit.text() == 'Dense':
                    self.net.add(Dense(units=self.structure.layers_list[i].spinBox.value(),
                                       kernel_initializer=k_initial,
                                       name='Dense_Layer_' + str(i)
                                       )
                                 )
                elif self.structure.layers_list[i].lineEdit.text() == 'Conv':
                    pass

                # add activation
                if activation == 'LeakyReLu':
                    self.net.add(LeakyReLU(alpha=0.3))
                elif activation == 'relu':
                    self.net.add(ReLU())
                elif activation == 'elu':
                    self.net.add(ELU())
                elif activation == 'softmax':
                    self.net.add(Softmax())

            self.net.add(Dense(units=self.y_input_size, kernel_initializer=tf.keras.initializers.RandomUniform(),
                               name='Output_Layer'))

            print(self.net.summary())

    def compile_model(self):
        global optimizer, net, epoches
        epoches = self.mWin.epoches
        self.remainder = epoches / 100
        optimizer = Adam(learning_rate=0.0002, )
        self.net.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        net = self.net

    def save_model(self, text):
        if text == 'struct':
            text, ok = QInputDialog.getText(self, '保存模型', '输入要保存模型的名字:')
            if text == '':
                QMessageBox.warning(self, '', '输入为空')
            elif ok and text:
                self.build_model_structure()
                self.model_name = path.dirname(path.realpath(sys.argv[0])) + '\\' + 'model' + '\\' + text + '.h5'
                self.net.save(self.model_name)
                print('加载模型:', self.model_name)
                QMessageBox.information(self, '', '模型保存成功')
                self.structure.treeWidget.topLevelItem(0).setText(1, '当前模型->' + self.model_name)
        else:
            if self.model_name == None:
                ok = QMessageBox.question(self, '', '需要先保存模型，是否保存当前的模型？', QMessageBox.Yes | QMessageBox.Cancel)
                if ok == QMessageBox.Yes:
                    self.save_model(text='struct')
                    self.save_model(text='weight')
            else:
                self.net.save(self.model_name)
                QMessageBox.information(self, '', '权重保存成功')

    def train_model(self):
        if self.dataset_name is not None:
            self.bn_continue_train.setEnabled(False)
            self.bn_predict.setEnabled(False)

            self.build_model_structure()
            self.compile_model()

            self.bn_train.setText('停止训练')
            self.bn_train.clicked.disconnect()
            self.bn_train.clicked.connect(self.stop_training)

            self.train_loss = [0.0]
            self.test_loss = [0.0]
            self.predict = [0.0]

            self.my_thread.is_on = True
            self.my_thread.start()  # 启动线程
        else:
            QMessageBox.warning(self, '', '未加载数据集')

    def continue_train(self):
        if self.dataset_name is not None:
            self.bn_train.setEnabled(False)
            self.bn_predict.setEnabled(False)

            self.compile_model()

            self.bn_continue_train.setText('停止训练')
            self.bn_continue_train.clicked.disconnect()
            self.bn_continue_train.clicked.connect(self.stop_training)

            self.predict = [0.0]

            self.my_thread.is_on = True
            self.my_thread.start()  # 启动线程
        else:
            QMessageBox.warning(self, '', '未加载数据集')

    def stop_training(self):
        self.bn_train.setText('从头训练')
        self.bn_train.clicked.disconnect()
        self.bn_train.clicked.connect(self.train_model)

        self.bn_continue_train.setText('继续训练')
        self.bn_continue_train.clicked.disconnect()
        self.bn_continue_train.clicked.connect(self.continue_train)

        self.my_thread.is_on = False

        self.bn_train.setEnabled(True)
        self.bn_continue_train.setEnabled(True)
        self.bn_saveWeight.setEnabled(True)
        self.progressBar.setValue(0)

    def train_end(self):
        self.stop_training()
        global net
        self.net = net

        self.bn_continue_train.setEnabled(True)
        self.bn_predict.setEnabled(True)

        QMessageBox.information(self, '', '训练完成')
        self.make_predict()

    def make_predict(self):
        if self.dataset_name is not None:
            self.predict = self.net(MinMaxScaler().fit_transform(self.x_data))
            # 必须实例化
            scale = MinMaxScaler()
            scale.fit_transform(y_data.reshape(-1, 1))

            self.predict = scale.inverse_transform(self.predict.numpy())
            self.predict = tf.squeeze(self.predict, axis=1)

            print('模型预测')
            QMessageBox.information(self, '', '预测结果可到“模型输出”查看')
            self.mWin.graph_widget.change_graph('预测曲线')
        else:
            QMessageBox.warning(self, '', '需要先加载数据集')
