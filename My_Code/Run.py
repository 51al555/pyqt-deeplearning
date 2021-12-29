import sys
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from demo import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeWidgetItem, QSpinBox, QComboBox, QLineEdit, QFileDialog, \
    QAction, QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtGui import QIcon
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, Concatenate, LeakyReLU
from tensorflow.keras.models import load_model
import time
import pandas as pd
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import add
from Modules_Run import PyQt5_And_TensorFlow_Demo
tf.compat.v1.disable_eager_execution()

# tensorflow和qt的线程类不合


class Tree_Item(QTreeWidgetItem):
    def __init__(self, tree):
        super(Tree_Item, self).__init__()
        self.tree = tree
        self.l1 = [QLineEdit(), QComboBox(), QSpinBox(), QComboBox(), QComboBox()]
        self.l1[1].setStyleSheet("background-color: #ffffff")
        self.l1[1].addItems(['Dense'])
        self.l1[1].setCurrentIndex(0)

        self.l1[2].setMaximum(1024)

        self.l1[3].addItems(['linear', 'elu', 'relu', 'LeakyReLu'])
        self.l1[3].setStyleSheet("background-color: #ffffff")

        self.l1[4].addItems(['RandomUniform', 'he_normal', 'he_uniform'])
        self.l1[4].setStyleSheet("background-color: #ffffff")

    def Tree_Item_add(self, index, root_item):
        root_item.insertChild(index, self)
        for i in range(len(self.l1)):
            self.tree.setItemWidget(self, i, self.l1[i])

    def Tree_Item_delete(self, root_item):
        root_item.removeChild(self)


class EmittingStr(QObject):
    textWritten = pyqtSignal(str)  # 定义一个发送str的信号

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


# class printThread(QThread):
#     stop_flag = False
#     update_i = pyqtSignal(int)
#
#     def run(self):
#         for i in range(101):
#             if self.stop_flag:
#                 return
#             self.update_i.emit(i)
#             time.sleep(0.1)
#         print("End")


class PyQt_Deeplearning_Demo(QMainWindow, Ui_MainWindow):
    tree_g_item_list = []
    tree_d_item_list = []
    tree_r_item_list = []
    temp = []
    index = 0

    def __init__(self):
        super(PyQt_Deeplearning_Demo, self).__init__()
        self.setupUi(self)
        self.initUI()
        self.model_state = True
        self.model_train_flag = False

        # user-defined output
        self.stdout_default = sys.stdout
        self.stderr_default = sys.stderr
        sys.stdout = EmittingStr(textWritten=self.outputWritten)
        sys.stderr = EmittingStr(textWritten=self.outputWritten)

        self.model = rCWGAN(self)
        self.bn_load_all.clicked.connect(self.model.load_all)
        self.bn_load_model_g.clicked.connect(partial(self.model.load_model, 'g_model_load'))
        self.bn_load_model_d.clicked.connect(partial(self.model.load_model, 'd_model_load'))
        self.bn_load_model_r.clicked.connect(partial(self.model.load_model, 'r_model_load'))
        self.bn_load_dataset.clicked.connect(self.model.load_dataset)

    def initUI(self):
        # dockwidget
        # self.dockWidget.hide()

        # menu
        terminal_show = QAction('显示', self)
        terminal_clear = QAction('清空', self)
        terminal_clear.setShortcut('ctrl+z')
        self.menubar.addAction(terminal_show)
        self.menubar.addSeparator()
        self.menubar.addAction(terminal_clear)

        info = QAction('使用信息', self)
        self.menubar_2.addAction(info)

        # treewidget
        self.treeWidget.setColumnWidth(0, 150)
        self.treeWidget.setColumnWidth(1, 100)
        self.treeWidget.setColumnWidth(2, 100)
        self.treeWidget.setColumnWidth(3, 100)
        self.treeWidget.expandAll()
        self.treeWidget.setFocus()
        self.treeWidget.setItemsExpandable(0)

        # comboxbox
        self.cb = QComboBox()

        # label
        self.dataset_path.setWordWrap(True)
        self.generator_path.setWordWrap(True)
        self.discriminator_path.setWordWrap(True)
        self.regressor_path.setWordWrap(True)

        # splitter
        # self.splitter.setStretchFactor(0, 3)
        # self.splitter.setStretchFactor(1, 7)
        #
        # self.splitter_2.setStretchFactor(0, 5)
        # self.splitter_2.setStretchFactor(1, 2)

        # textBrowser
        self.toolBox.setStyleSheet("background-color: #f0f0f0")

        # signals and slots
        self.bn_train.clicked.connect(self.Model_run_or_stop)
        self.toolBox.currentChanged.connect(self.toolBoxChanged)
        self.bn_plus.clicked.connect(self.node_add)
        self.bn_minus.clicked.connect(self.node_delete)
        self.bn_go_up.clicked.connect(self.node_up)
        self.bn_go_down.clicked.connect(self.node_down)
        terminal_show.triggered.connect(self.menu_clicked)
        terminal_clear.triggered.connect(self.menu_clicked)
        info.triggered.connect(self.info_clicked)

        # images
        self.bn_plus.setIcon(QIcon('./material/plus.png'))
        self.bn_minus.setIcon(QIcon('./material/minus.png'))
        self.bn_go_up.setIcon(QIcon('./material/go_up.png'))
        self.bn_go_down.setIcon(QIcon('./material/go_down.png'))

        # thread
        # self.t = printThread()
        # self.t.update_i.connect(self.thread_update_UI)

        self.node_flush(init=True)

    def node_add(self):
        try:
            self.treeWidget.currentItem().parent().parent()
            item = self.treeWidget.currentItem().parent()
            self.index = self.treeWidget.currentIndex().row() + 1  # 输入层:1
        except:
            item = self.treeWidget.currentItem()
            if item.text(0) == '生成器':
                self.index = len(self.tree_g_item_list)
            elif item.text(0) == '判别器':
                self.index = len(self.tree_d_item_list)
            elif item.text(0) == '回归器':
                self.index = len(self.tree_r_item_list)
        if item.text(0) == '生成器':
            self.tree_g_item_list.insert(self.index, Tree_Item(self.treeWidget))
            self.tree_g_item_list[self.index].Tree_Item_add(self.index, item)
            self.tree_g_item_list[self.index].l1[0].setText('Hidden Layer')
            print('生成器->添加节点' + str(self.index + 1))
        elif item.text(0) == '判别器':
            self.tree_d_item_list.insert(self.index, Tree_Item(self.treeWidget))
            self.tree_d_item_list[self.index].Tree_Item_add(self.index, item)
            self.tree_d_item_list[self.index].l1[0].setText('Hidden Layer')
            print('判别器->添加节点' + str(self.index + 1))
        elif item.text(0) == '回归器':
            self.tree_r_item_list.insert(self.index, Tree_Item(self.treeWidget))
            self.tree_r_item_list[self.index].Tree_Item_add(self.index, item)
            self.tree_r_item_list[self.index].l1[0].setText('Hidden Layer')
            print('回归器->添加节点' + str(self.index + 1))

    def node_delete(self):
        try:
            self.treeWidget.currentItem().parent().parent()
            item = self.treeWidget.currentItem().parent()
            self.index = self.treeWidget.currentIndex().row()
            if item.text(0) == '生成器':
                self.tree_g_item_list[self.index].Tree_Item_delete(item)
                self.tree_g_item_list.pop(self.index)
                print('生成器->删除节点' + str(self.index + 1))
            elif item.text(0) == '判别器':
                self.tree_d_item_list[self.index].Tree_Item_delete(item)
                self.tree_d_item_list.pop(self.index)
                print('判别器->删除节点' + str(self.index + 1))
            elif item.text(0) == '回归器':
                self.tree_r_item_list[self.index].Tree_Item_delete(item)
                self.tree_r_item_list.pop(self.index)
                print('回归器->删除节点' + str(self.index + 1))
        except:
            item = self.treeWidget.currentItem()
            if item.text(0) == '生成器':
                if len(self.tree_g_item_list) != 0:
                    self.tree_g_item_list[-1].Tree_Item_delete(item)
                    self.tree_g_item_list.pop()
                    print('生成器->删除节点')
            elif item.text(0) == '判别器':
                if len(self.tree_d_item_list) != 0:
                    self.tree_d_item_list[-1].Tree_Item_delete(item)
                    self.tree_d_item_list.pop()
                    print('判别器->删除节点')
            elif item.text(0) == '回归器':
                if len(self.tree_r_item_list) != 0:
                    self.tree_r_item_list[-1].Tree_Item_delete(item)
                    self.tree_r_item_list.pop()
                    print('回归器->删除节点')

    def node_up(self):
        try:
            self.treeWidget.currentItem().parent().parent()
            item = self.treeWidget.currentItem().parent()
            self.index = self.treeWidget.currentIndex().row()
            if self.index != 0:
                if item.text(0) == '生成器':
                    self.tree_g_item_list[self.index-1], self.tree_g_item_list[self.index] = self.tree_g_item_list[self.index], self.tree_g_item_list[self.index-1]
                    self.node_flush()
                    self.treeWidget.setCurrentItem(self.tree_g_item_list[self.index-1])
                elif item.text(0) == '判别器':
                    self.tree_d_item_list[self.index-1], self.tree_d_item_list[self.index] = self.tree_d_item_list[self.index], self.tree_d_item_list[self.index-1]
                    self.node_flush()
                    self.treeWidget.setCurrentItem(self.tree_d_item_list[self.index-1])
                elif item.text(0) == '回归器':
                    self.tree_r_item_list[self.index-1], self.tree_r_item_list[self.index] = self.tree_r_item_list[self.index], self.tree_r_item_list[self.index-1]
                    self.node_flush()
                    self.treeWidget.setCurrentItem(self.tree_r_item_list[self.index-1])
        except:
            pass

    def node_down(self):
        try:
            self.treeWidget.currentItem().parent().parent()
            item = self.treeWidget.currentItem().parent()
            self.index = self.treeWidget.currentIndex().row()
            if item.text(0) == '生成器':
                if self.index != (len(self.tree_g_item_list) - 1):
                    self.tree_g_item_list[self.index+1], self.tree_g_item_list[self.index] = self.tree_g_item_list[self.index], self.tree_g_item_list[self.index+1]
                    self.node_flush()
                    self.treeWidget.setCurrentItem(self.tree_g_item_list[self.index+1])
            elif item.text(0) == '判别器':
                if self.index != (len(self.tree_d_item_list) - 1):
                    self.tree_d_item_list[self.index+1], self.tree_d_item_list[self.index] = self.tree_d_item_list[self.index], self.tree_d_item_list[self.index+1]
                    self.node_flush()
                    self.treeWidget.setCurrentItem(self.tree_d_item_list[self.index+1])
            elif item.text(0) == '回归器':
                if self.index != (len(self.tree_r_item_list) - 1):
                    self.tree_r_item_list[self.index+1], self.tree_r_item_list[self.index] = self.tree_r_item_list[self.index], self.tree_r_item_list[self.index+1]
                    self.node_flush()
                    self.treeWidget.setCurrentItem(self.tree_r_item_list[self.index+1])
        except:
            pass

    def node_flush(self, init=False):
        if init:
            self.tree_g_item_list.append(Tree_Item(self.treeWidget))
            self.tree_g_item_list[0].Tree_Item_add(0, self.treeWidget.topLevelItem(0))
            self.tree_g_item_list[0].l1[0].setText('Hidden Layer')

            self.tree_d_item_list.append(Tree_Item(self.treeWidget))
            self.tree_d_item_list[0].Tree_Item_add(0, self.treeWidget.topLevelItem(1))
            self.tree_d_item_list[0].l1[0].setText('Hidden Layer')

            self.tree_r_item_list.append(Tree_Item(self.treeWidget))
            self.tree_r_item_list[0].Tree_Item_add(0, self.treeWidget.topLevelItem(2))
            self.tree_r_item_list[0].l1[0].setText('Hidden Layer')
            return

        try:
            self.treeWidget.currentItem().parent().parent()
            item = self.treeWidget.currentItem().parent()
            if item.text(0) == '生成器':
                # 暂且这么写，循环前套始终报错，内存地址
                for i in range(len(self.tree_g_item_list)):
                    # 创建新对象，逐一赋值
                    self.tree_g_item_list[i].Tree_Item_delete(self.treeWidget.topLevelItem(0))
                    self.temp.append(Tree_Item(self.treeWidget))
                    self.temp[i].l1[0].setText(self.tree_g_item_list[i].l1[0].text())
                    self.temp[i].l1[1].setCurrentIndex(self.tree_g_item_list[i].l1[1].currentIndex())
                    self.temp[i].l1[2].setValue(self.tree_g_item_list[i].l1[2].value())
                    self.temp[i].l1[3].setCurrentIndex(self.tree_g_item_list[i].l1[3].currentIndex())
                    self.temp[i].l1[4].setCurrentIndex(self.tree_g_item_list[i].l1[4].currentIndex())
                self.tree_g_item_list = []  # 清空对象
                for i in range(len(self.temp)):
                    self.temp[i].Tree_Item_add(i, self.treeWidget.topLevelItem(0))
                self.tree_g_item_list = self.temp[:]  # 对象复制
                self.temp = []  # 清空对象
            elif item.text(0) == '判别器':
                for i in range(len(self.tree_d_item_list)):
                    self.tree_d_item_list[i].Tree_Item_delete(self.treeWidget.topLevelItem(1))
                    self.temp.append(Tree_Item(self.treeWidget))
                    self.temp[i].l1[0].setText(self.tree_d_item_list[i].l1[0].text())
                    self.temp[i].l1[1].setCurrentIndex(self.tree_d_item_list[i].l1[1].currentIndex())
                    self.temp[i].l1[2].setValue(self.tree_d_item_list[i].l1[2].value())
                    self.temp[i].l1[3].setCurrentIndex(self.tree_d_item_list[i].l1[3].currentIndex())
                    self.temp[i].l1[4].setCurrentIndex(self.tree_d_item_list[i].l1[4].currentIndex())
                self.tree_d_item_list = []  # 清空对象
                for i in range(len(self.temp)):
                    self.temp[i].Tree_Item_add(i, self.treeWidget.topLevelItem(1))
                self.tree_d_item_list = self.temp[:]  # 对象复制
                self.temp = []  # 清空对象
            elif item.text(0) == '回归器':
                for i in range(len(self.tree_r_item_list)):
                    self.tree_r_item_list[i].Tree_Item_delete(self.treeWidget.topLevelItem(2))
                    self.temp.append(Tree_Item(self.treeWidget))
                    self.temp[i].l1[0].setText(self.tree_r_item_list[i].l1[0].text())
                    self.temp[i].l1[1].setCurrentIndex(self.tree_r_item_list[i].l1[1].currentIndex())
                    self.temp[i].l1[2].setValue(self.tree_r_item_list[i].l1[2].value())
                    self.temp[i].l1[3].setCurrentIndex(self.tree_r_item_list[i].l1[3].currentIndex())
                    self.temp[i].l1[4].setCurrentIndex(self.tree_r_item_list[i].l1[4].currentIndex())
                self.tree_r_item_list = []  # 清空对象
                for i in range(len(self.temp)):
                    self.temp[i].Tree_Item_add(i, self.treeWidget.topLevelItem(2))
                self.tree_r_item_list = self.temp[:]  # 对象复制
                self.temp = []  # 清空对象
        except:
            pass

    def toolBoxChanged(self, index):
        # if index == 0:
        #     self.model = rCWGAN(g_args=self.tree_g_item_list, d_args=self.tree_d_item_list, r_args=self.tree_r_item_list)
        # elif index == 1:
        #     self.model = WGAN_GP()
        # print('toolBoxChanged:', index)
        pass

    def Model_run_or_stop(self):
        if self.model.dataset_flag:
            if self.model_state:
                self.model.g_args = self.tree_g_item_list
                self.model.d_args = self.tree_d_item_list
                self.model.r_args = self.tree_r_item_list

                self.model.init_model()
                self.model.compile_model()
                self.model_state = False
                self.model_train_flag = True
                # self.t.start()
                self.bn_train.setText('停止训练')
                self.model.train_model()

            else:
                self.model_state = True
                self.model_train_flag = False
                self.bn_train.setText('开始训练')
            self.model_state = not self.model_state
        else:
            QMessageBox.warning(self, '提示', '未加载数据集', QMessageBox.Yes)
    # def thread_update_UI(self, i):
    #     self.progressBar.setValue(i)
        # self.textEdit.append(i)

    def outputWritten(self, text):
        self.textEdit.insertPlainText(text)
        self.textEdit.moveCursor(self.textEdit.textCursor().End)

    def menu_clicked(self):
        if self.sender().text() == '显示':
            self.dockWidget.show()
        elif self.sender().text() == '清空':
            self.textEdit.clear()

    def info_clicked(self):
        if self.sender().text() == '使用信息':
            QMessageBox.information(self, '工具箱使用信息', '1、已加载模型的网络默认继续训练，未加载模型的网络默认按照'
                                                     '自定义结构从零开始训练。\n\n2、选择“继续训练”，未改变结构的网络在'
                                                     '上一次基础上继续进行训练，改变结构的网络从起始状态开始训练。')


class rCWGAN():
    activation = None
    generator_model_flag = False
    discriminator_model_flag = False
    regressor_model_flag = False
    dataset_flag = False
    g_args = []
    d_args = []
    r_args = []
    batch_size = 35
    optim_disc = Adam(learning_rate=0.0003, beta_1=0, beta_2=0.9)
    optim_reg = Adam(learning_rate=0.0005,)
    optim_gen = Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)

    def __init__(self, mainwin):
        self.mainwin = mainwin

        # 真实数据集
        self.x_input_size = 17
        self.y_input_size = 1

        # 噪声
        self.z_input_size = 10

    def load_dataset(self, isloadAll):
        if isloadAll:
            fname = sys.path[0] + '\\' + 'PTA_DATA.xls'
        else:
            fname, _ = QFileDialog.getOpenFileName(self.mainwin, '选择数据集文件', '.', '数据集文件(*.xls)')
            if fname == '':
                return
        self.mainwin.dataset_path.setText(fname)
        data = pd.read_excel(fname, header=None)
        data = data.values
        self.x = data[:, :17]
        self.y = data[:, 17]
        self.dataset_flag = True

    def load_model(self, text, isloadAll):
        if isloadAll:
            fname = sys.path[0] + '\\' + 'g.h5'
            self.generator_model = load_model(fname)
            self.generator_model.summary()
            self.mainwin.treeWidget.topLevelItem(0).setDisabled(1)
            self.mainwin.treeWidget.topLevelItem(0).setExpanded(0)
            self.mainwin.generator_path.setText(fname)
            self.generator_model_flag = True

            fname = sys.path[0] + '\\' + 'd.h5'
            self.discriminator_model = load_model(fname)
            self.discriminator_model.summary()
            self.mainwin.treeWidget.topLevelItem(1).setDisabled(1)
            self.mainwin.treeWidget.topLevelItem(1).setExpanded(0)
            self.mainwin.discriminator_path.setText(fname)
            self.discriminator_model_flag = True

            fname = sys.path[0] + '\\' + 'r.h5'
            self.regressor_model = load_model(fname)
            self.regressor_model.summary()
            self.mainwin.treeWidget.topLevelItem(2).setDisabled(1)
            self.mainwin.treeWidget.topLevelItem(2).setExpanded(0)
            self.mainwin.regressor_path.setText(fname)
            self.regressor_model_flag = True
        else:
            fname, _ = QFileDialog.getOpenFileName(self.mainwin, '选择模型文件', '.', '模型文件(*.h5 *.tflite)')
            if fname == '':
                return
            if text == 'g_model_load':
                self.generator_model = load_model(fname)
                self.generator_model.summary()
                self.mainwin.treeWidget.topLevelItem(0).setDisabled(1)
                self.mainwin.treeWidget.topLevelItem(0).setExpanded(0)
                self.mainwin.generator_path.setText(fname)
                self.generator_model_flag = True
            elif text == 'd_model_load':
                self.discriminator_model = load_model(fname)
                self.discriminator_model.summary()
                self.mainwin.treeWidget.topLevelItem(1).setDisabled(1)
                self.mainwin.treeWidget.topLevelItem(1).setExpanded(0)
                self.mainwin.discriminator_path.setText(fname)
                self.discriminator_model_flag = True
            elif text == 'r_model_load':
                self.regressor_model = load_model(fname)
                self.regressor_model.summary()
                self.mainwin.treeWidget.topLevelItem(2).setDisabled(1)
                self.mainwin.treeWidget.topLevelItem(2).setExpanded(0)
                self.mainwin.regressor_path.setText(fname)
                self.regressor_model_flag = True

    def load_all(self):
        self.load_dataset(isloadAll=True)
        self.load_model(text='', isloadAll=True)

    def init_model(self):
        # G
        if self.generator_model_flag:
            print('generator model is loaded')
        else:
            noise_z = Input(shape=(self.z_input_size,), dtype=float, name='Generator_input_noise_z')
            label_y = Input(shape=(self.y_input_size,), dtype=float, name='Generator_input_label_y')
            d = Concatenate()([noise_z, label_y])   # Input Layer

            for i in range(len(self.g_args)):
                self.activation = self.g_args[i].l1[3].currentText()
                if self.activation == 'LeakyReLu':
                    self.activation = LeakyReLU(alpha=0.2)
                d = Dense(units=self.g_args[i].l1[2].value(), activation=self.activation)(d)
            m_output = Dense(units=self.x_input_size)(d)
            self.generator_model = Model(inputs=[noise_z, label_y], outputs=m_output)
            self.generator_model.save('g2.h5')
        # D
        if self.discriminator_model_flag:
            print('discriminator model is loaded')
        else:
            label = Input(shape=(self.y_input_size,), dtype=float, name='Discriminator_input_y')
            d_x = Input(shape=(self.x_input_size,), dtype=float, name='Discriminator_input_x')
            d = Concatenate()([d_x, label])   # Input Layer

            for i in range(len(self.d_args)):
                self.activation = self.d_args[i].l1[3].currentText()
                if self.activation == 'LeakyReLu':
                    self.activation = LeakyReLU(alpha=0.2)
                d = Dense(units=self.d_args[i].l1[2].value(), activation=self.activation)(d)
            valid = Dense(units=1, activation="sigmoid")(d)
            self.discriminator_model = Model(inputs=[d_x, label], outputs=valid)
            self.discriminator_model.save('d2.h5')
        # R
        if self.regressor_model_flag:
            print('regressor model is loaded')
        else:
            r_x = Input(shape=(self.x_input_size,), dtype=float, name='Regressor_input')

            self.activation = self.r_args[i].l1[3].currentText()
            if self.activation == 'LeakyReLu':
                self.activation = LeakyReLU(alpha=0.2)
            d = Dense(units=self.r_args[0].l1[2].value(), activation=self.activation)(r_x)

            for i in range(len(self.r_args)):
                if i != 0:
                    self.activation = self.r_args[i].l1[3].currentText()
                    if self.activation == 'LeakyReLu':
                        self.activation = LeakyReLU(alpha=0.2)
                    d = Dense(units=self.r_args[i].l1[2].value(), activation=self.activation)(d)
            pre_y = Dense(units=self.y_input_size, activation="linear")(d)
            self.regressor_model = Model(inputs=r_x, outputs=pre_y)
            self.regressor_model.save('r2.h5')

    def RandomWeightAverage(self, inputs):
        alpha = K.random_uniform(shape=(self.batch_size, 1), minval=0, maxval=1)
        inter = add([alpha*inputs[0], (1-alpha)*inputs[1]])
        return inter

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples, in_label):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([averaged_samples, in_label])
            y_pred = self.discriminator_model([averaged_samples, in_label])
            gradient = tape.gradient(y_pred, [averaged_samples, in_label])
            gradient = tf.concat([gradient[0], gradient[1]], axis=-1)
            gradient_L2_norm = tf.norm(gradient, ord=2, axis=1)
            gradient_penalty = K.square(gradient_L2_norm - 1)
        return gradient_penalty

    def wassertein_loss(self, y_true, y_pred):
        a = K.mean(y_true * y_pred)
        return a

    def compile_model(self):
        self.regressor_model.compile(self.optim_reg, loss='mse')
        self.generator_model.trainable = False

        x = Input(shape=(self.x_input_size,), name="real_x")
        y = Input(shape=(self.y_input_size,), name="label")
        fake_y = Input(shape=(self.y_input_size,), name="fake_label")
        z = Input(shape=(self.z_input_size,))

        fake_x = self.generator_model([z, fake_y])
        fake = self.discriminator_model([fake_x, fake_y])
        valid = self.discriminator_model([x, y])

        interpolated_data = self.RandomWeightAverage([x, fake_x])
        interpolated_value = self.discriminator_model([interpolated_data, y])

        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=interpolated_data, in_label=y)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.discriminator_model_2 = Model(inputs=[x, z, y, fake_y], outputs=[valid, fake, interpolated_value])
        self.discriminator_model_2.compile(optimizer=self.optim_disc,
                                           loss=[self.wassertein_loss, self.wassertein_loss, partial_gp_loss],
                                           loss_weights=[1, 1, 10], experimental_run_tf_function=False)

        self.discriminator_model.trainable = False
        self.generator_model.trainable = True

        z_gen = Input(shape=(self.z_input_size,), name="z_gen")
        label_gen = Input(shape=(self.y_input_size,), name="label_gen")
        g_z = self.generator_model([z_gen, label_gen])
        valid = self.discriminator_model([g_z, label_gen])
        self.generator_model_2 = Model([z_gen, label_gen], valid)
        self.generator_model_2.compile(optimizer=self.optim_gen, loss=self.wassertein_loss)

    def train_model(self):
        x_train, x_test = train_test_split(self.x, test_size=0.3, random_state=1)
        y_train, y_test = train_test_split(self.y, test_size=0.3, random_state=1)

        # normalization
        x_train = MinMaxScaler().fit_transform(x_train)
        y_train = MinMaxScaler().fit_transform(y_train.reshape(-1, 1))
        x_test = MinMaxScaler().fit_transform(x_test)

        epochs = 100
        critic_num = 3

        valid = -np.ones((self.batch_size, 1))
        fake = np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1))
        dLossErr = np.zeros([epochs, 1])
        gLossErr = np.zeros([epochs, 1])
        rLossErr1 = np.zeros([epochs, 1])
        rLossErr2 = np.zeros([epochs, 1])
        for epoch in range(epochs):
            if self.mainwin.model_train_flag:
                for _ in range(critic_num):
                    if self.mainwin.model_train_flag:
                        idx = np.random.randint(0, x_train.shape[0], size=self.batch_size)
                        x = x_train[idx]
                        y = y_train[idx]
                        noise = tf.random.uniform(shape=(self.batch_size, 10))
                        z_g = self.generator_model.predict_on_batch([noise, y])
                        QApplication.processEvents()
                        r_label = self.regressor_model.predict_on_batch(z_g)
                        QApplication.processEvents()
                        r_loss1 = self.regressor_model.train_on_batch(x, y)
                        QApplication.processEvents()
                        r_loss2 = self.regressor_model.train_on_batch(z_g, y)
                        QApplication.processEvents()
                        d_loss = self.discriminator_model_2.train_on_batch([x, noise, y, r_label], [valid, fake, dummy])
                        QApplication.processEvents()

                g_loss = self.generator_model_2.train_on_batch([noise, y], valid)
                QApplication.processEvents()
                dLossErr[epoch] = d_loss[0]
                gLossErr[epoch] = g_loss
                rLossErr1[epoch] = r_loss1
                rLossErr2[epoch] = r_loss2
                str_emit = "====[Epoch: %d/%d] [D loss: %f] [G loss: %f],[R loss1:  %f],[R loss2:  %f]" % (
                epoch + 1, epochs, d_loss[0], g_loss, r_loss1, r_loss2)
                self.mainwin.textEdit.append(str_emit)
                self.mainwin.textEdit.moveCursor(self.mainwin.textEdit.textCursor().End)
                self.mainwin.progressBar.setValue(epoch+1)

                # print("====[Epoch: %d/%d] [D loss: %f] [G loss: %f],[R loss1:  %f],[R loss2:  %f]" % (
                # epoch + 1, epochs, d_loss[0], g_loss, r_loss1, r_loss2))
        self.mainwin.Model_run_or_stop()



class WGAN_GP():
    def __init__(self):
        pass


def exceptOutConfig(exctype, value, tb):
    print('Error Information:')
    print('Type:', exctype)
    print('Value:', value)
    print('Traceback:', tb)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = PyQt_Deeplearning_Demo()
    demo.show()
    demo2 = PyQt5_And_TensorFlow_Demo()
    sys.excepthook = exceptOutConfig
    sys.exit(app.exec_())
