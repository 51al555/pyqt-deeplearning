from functools import partial
import sys
from pandas import read_excel
import torch
import torch.nn as nn
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QHeaderView, QDialog, QTreeWidgetItem, QFileDialog
import UI_rcwgan
import UI_regressor
from Shared_Api import check_dataset, Layers_Choose, Layer_Dense
from os import path


class rCWGAN(QWidget, UI_rcwgan.Ui_Form):
    layer_selection = 'Dense'
    epoch = 500

    g_ui_list = []
    g_layers_list = []

    d_ui_list = []
    d_layers_list = []

    r_ui_list = []
    r_layers_list = []

    currentIndex = 0

    def __init__(self, app):
        super(rCWGAN, self).__init__()
        self.app = app
        self.setupUi(self)

        self.initUI()

    def initUI(self):
        self.treeWidget.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.treeWidget.expandAll()
        self.treeWidget.setCurrentItem(self.treeWidget.topLevelItem(0))

        # images
        self.bn_plus.setIcon(QIcon('./material/plus.png'))
        self.bn_minus.setIcon(QIcon('./material/minus.png'))
        self.bn_go_up.setIcon(QIcon('./material/go_up.png'))
        self.bn_go_down.setIcon(QIcon('./material/go_down.png'))

        # slot and signal
        self.bn_plus.clicked.connect(self.layer_add)
        self.bn_minus.clicked.connect(self.layer_delete)
        self.bn_go_up.clicked.connect(self.layer_go_up)
        self.bn_go_down.clicked.connect(self.layer_go_down)
        self.bn_load_all.clicked.connect(partial(self.load_models_and_dataset, 'All'))
        self.bn_load_model_g.clicked.connect(partial(self.load_models_and_dataset, 'Generator'))
        self.bn_load_model_d.clicked.connect(partial(self.load_models_and_dataset, 'Discriminator'))
        self.bn_load_model_r.clicked.connect(partial(self.load_models_and_dataset, 'Regressor'))
        self.bn_load_dataset.clicked.connect(partial(self.load_models_and_dataset, 'Dataset'))
        self.bn_check_dataset.clicked.connect(check_dataset)
        self.bn_train.clicked.connect(self.build_model_structure)

    def layer_add(self):
        dialog = Layers_Choose()
        self.layer_selection = 'Dense'
        dialog.comboBox.currentTextChanged.connect(self.dialog_selection)
        res = dialog.exec_()
        if res == QDialog.Accepted:
            if self.layer_selection == 'Dense':
                if self.treeWidget.currentItem().parent() is not None:
                    self.currentIndex = self.treeWidget.currentIndex().row() + 1
                    if self.treeWidget.currentItem().parent().text(0) == '生成器':
                        self.g_layers_list.insert(self.currentIndex, Layer_Dense())

                        # UI
                        self.g_ui_list.insert(self.currentIndex, QTreeWidgetItem())
                        self.treeWidget.topLevelItem(0).insertChild(self.currentIndex,
                                                                    self.g_ui_list[self.currentIndex])
                        self.treeWidget.setItemWidget(self.g_ui_list[self.currentIndex], 1,
                                                      self.g_layers_list[self.currentIndex])
                        print(self.treeWidget.currentItem().parent().text(0) + '->添加隐藏层')

                    elif self.treeWidget.currentItem().parent().text(0) == '判别器':
                        self.d_layers_list.insert(self.currentIndex, Layer_Dense())

                        # UI
                        self.d_ui_list.insert(self.currentIndex, QTreeWidgetItem())
                        self.treeWidget.topLevelItem(1).insertChild(self.currentIndex,
                                                                    self.d_ui_list[self.currentIndex])
                        self.treeWidget.setItemWidget(self.d_ui_list[self.currentIndex], 1,
                                                      self.d_layers_list[self.currentIndex])
                        print(self.treeWidget.currentItem().parent().text(0) + '->添加隐藏层')

                    elif self.treeWidget.currentItem().parent().text(0) == '回归器':
                        self.r_layers_list.insert(self.currentIndex, Layer_Dense())

                        # UI
                        self.r_ui_list.insert(self.currentIndex, QTreeWidgetItem())
                        self.treeWidget.topLevelItem(2).insertChild(self.currentIndex,
                                                                    self.r_ui_list[self.currentIndex])
                        self.treeWidget.setItemWidget(self.r_ui_list[self.currentIndex], 1,
                                                      self.r_layers_list[self.currentIndex])
                        print(self.treeWidget.currentItem().parent().text(0) + '->添加隐藏层')
                else:
                    if self.treeWidget.currentItem().text(0) == '生成器':
                        self.g_layers_list.append(Layer_Dense())

                        # UI
                        self.g_ui_list.append(QTreeWidgetItem())
                        self.treeWidget.topLevelItem(0).addChild(self.g_ui_list[-1])
                        self.treeWidget.setItemWidget(self.g_ui_list[-1], 1, self.g_layers_list[-1])
                        print(self.treeWidget.currentItem().text(0) + '->添加隐藏层')

                    elif self.treeWidget.currentItem().text(0) == '判别器':
                        self.d_layers_list.append(Layer_Dense())

                        # UI
                        self.d_ui_list.append(QTreeWidgetItem())
                        self.treeWidget.topLevelItem(1).addChild(self.d_ui_list[-1])
                        self.treeWidget.setItemWidget(self.d_ui_list[-1], 1, self.d_layers_list[-1])
                        print(self.treeWidget.currentItem().text(0) + '->添加隐藏层')

                    elif self.treeWidget.currentItem().text(0) == '回归器':
                        self.r_layers_list.append(Layer_Dense())

                        # UI
                        self.r_ui_list.append(QTreeWidgetItem())
                        self.treeWidget.topLevelItem(2).addChild(self.r_ui_list[-1])
                        self.treeWidget.setItemWidget(self.r_ui_list[-1], 1, self.r_layers_list[-1])
                        print(self.treeWidget.currentItem().text(0) + '->添加隐藏层')

            elif self.layer_selection == 'Dropout':
                print(self.layer_selection)
            elif self.layer_selection == 'Conv':
                print(self.layer_selection)

    def layer_delete(self):
        if self.treeWidget.currentItem().parent() is not None:
            self.currentIndex = self.treeWidget.currentIndex().row()
            if self.treeWidget.currentItem().parent().text(0) == '生成器':
                self.g_layers_list.pop(self.currentIndex)

                # UI
                self.treeWidget.topLevelItem(0).removeChild(self.g_ui_list[self.currentIndex])
                self.g_ui_list.pop(self.currentIndex)
                print(self.treeWidget.currentItem().parent().text(0) + '->删除隐藏层')

            elif self.treeWidget.currentItem().parent().text(0) == '判别器':
                self.d_layers_list.pop(self.currentIndex)

                # UI
                self.treeWidget.topLevelItem(1).removeChild(self.d_ui_list[self.currentIndex])
                self.d_ui_list.pop(self.currentIndex)
                print(self.treeWidget.currentItem().parent().text(0) + '->删除隐藏层')

            elif self.treeWidget.currentItem().parent().text(0) == '回归器':
                self.r_layers_list.pop(self.currentIndex)

                # UI
                self.treeWidget.topLevelItem(2).removeChild(self.r_ui_list[self.currentIndex])
                self.r_ui_list.pop(self.currentIndex)
                print(self.treeWidget.currentItem().parent().text(0) + '->删除隐藏层')
        else:
            if self.treeWidget.currentItem().text(0) == '生成器' and (len(self.g_layers_list) != 0):
                self.g_layers_list.pop()

                # UI
                self.treeWidget.topLevelItem(0).removeChild(self.g_ui_list[-1])
                self.g_ui_list.pop()
                print(self.treeWidget.currentItem().text(0) + '->删除隐藏层')

            elif self.treeWidget.currentItem().text(0) == '判别器' and (len(self.d_layers_list) != 0):
                self.d_layers_list.pop()

                # UI
                self.treeWidget.topLevelItem(1).removeChild(self.d_ui_list[-1])
                self.d_ui_list.pop()
                print(self.treeWidget.currentItem().text(0) + '->删除隐藏层')

            elif self.treeWidget.currentItem().text(0) == '回归器' and (len(self.r_layers_list) != 0):
                self.r_layers_list.pop()

                # UI
                self.treeWidget.topLevelItem(2).removeChild(self.r_ui_list[-1])
                self.r_ui_list.pop()
                print(self.treeWidget.currentItem().text(0) + '->删除隐藏层')

    def layer_flush(self, txt):
        if txt == '生成器':
            g_layer_list = []
            for i in range(len(self.g_layers_list)):
                g_layer_list.append(Layer_Dense())
                g_layer_list[i].copy(self.g_layers_list[i])

            self.treeWidget.topLevelItem(0).takeChildren()
            for i in range(len(self.g_layers_list)):
                self.treeWidget.topLevelItem(0).addChild(self.g_ui_list[i])
                self.treeWidget.setItemWidget(self.g_ui_list[i], 1, g_layer_list[i])

            self.g_layers_list = g_layer_list[:]

        elif txt == '判别器':
            d_layer_list = []
            for i in range(len(self.d_layers_list)):
                d_layer_list.append(Layer_Dense())
                d_layer_list[i].copy(self.d_layers_list[i])

            self.treeWidget.topLevelItem(1).takeChildren()
            for i in range(len(self.d_layers_list)):
                self.treeWidget.topLevelItem(1).addChild(self.d_ui_list[i])
                self.treeWidget.setItemWidget(self.d_ui_list[i], 1, d_layer_list[i])

            self.d_layers_list = d_layer_list[:]

        elif txt == '回归器':
            r_layer_list = []
            for i in range(len(self.r_layers_list)):
                r_layer_list.append(Layer_Dense())
                r_layer_list[i].copy(self.r_layers_list[i])

            self.treeWidget.topLevelItem(2).takeChildren()
            for i in range(len(self.r_layers_list)):
                self.treeWidget.topLevelItem(2).addChild(self.r_ui_list[i])
                self.treeWidget.setItemWidget(self.r_ui_list[i], 1, r_layer_list[i])

            self.r_layers_list = r_layer_list[:]

    def layer_go_up(self):
        if self.treeWidget.currentItem().parent() is not None:
            self.currentIndex = self.treeWidget.currentIndex().row()
            if self.currentIndex != 0:
                if self.treeWidget.currentItem().parent().text(0) == '生成器':
                    self.g_layers_list[self.currentIndex - 1], self.g_layers_list[self.currentIndex] = \
                        self.g_layers_list[
                            self.currentIndex], \
                        self.g_layers_list[
                            self.currentIndex - 1]
                    self.layer_flush('生成器')
                    self.treeWidget.setCurrentItem(self.g_ui_list[self.currentIndex - 1])
                    print(self.treeWidget.currentItem().parent().text(0) + '->上移当前隐藏层')

                elif self.treeWidget.currentItem().parent().text(0) == '判别器':
                    self.d_layers_list[self.currentIndex - 1], self.d_layers_list[self.currentIndex] = \
                        self.d_layers_list[
                            self.currentIndex], \
                        self.d_layers_list[
                            self.currentIndex - 1]
                    self.layer_flush('判别器')
                    self.treeWidget.setCurrentItem(self.d_ui_list[self.currentIndex - 1])
                    print(self.treeWidget.currentItem().parent().text(0) + '->上移当前隐藏层')

                elif self.treeWidget.currentItem().parent().text(0) == '回归器':
                    self.r_layers_list[self.currentIndex - 1], self.r_layers_list[self.currentIndex] = \
                        self.r_layers_list[
                            self.currentIndex], \
                        self.r_layers_list[
                            self.currentIndex - 1]
                    self.layer_flush('回归器')
                    self.treeWidget.setCurrentItem(self.r_ui_list[self.currentIndex - 1])
                    print(self.treeWidget.currentItem().parent().text(0) + '->上移当前隐藏层')

    def layer_go_down(self):
        if self.treeWidget.currentItem().parent() is not None:
            self.currentIndex = self.treeWidget.currentIndex().row()
            if self.treeWidget.currentItem().parent().text(0) == '生成器':
                if self.currentIndex != (len(self.g_layers_list) - 1):
                    self.g_layers_list[self.currentIndex + 1], self.g_layers_list[self.currentIndex] = \
                        self.g_layers_list[
                            self.currentIndex], \
                        self.g_layers_list[
                            self.currentIndex + 1]
                    self.layer_flush('生成器')
                    self.treeWidget.setCurrentItem(self.g_ui_list[self.currentIndex + 1])
                    print(self.treeWidget.currentItem().parent().text(0) + '->下移当前隐藏层')

            elif self.treeWidget.currentItem().parent().text(0) == '判别器':
                if self.currentIndex != (len(self.d_layers_list) - 1):
                    self.d_layers_list[self.currentIndex + 1], self.d_layers_list[self.currentIndex] = \
                        self.d_layers_list[
                            self.currentIndex], \
                        self.d_layers_list[
                            self.currentIndex + 1]
                    self.layer_flush('判别器')
                    self.treeWidget.setCurrentItem(self.d_ui_list[self.currentIndex + 1])
                    print(self.treeWidget.currentItem().parent().text(0) + '->下移当前隐藏层')

            elif self.treeWidget.currentItem().parent().text(0) == '回归器':
                if self.currentIndex != (len(self.r_layers_list) - 1):
                    self.r_layers_list[self.currentIndex + 1], self.r_layers_list[self.currentIndex] = \
                        self.r_layers_list[
                            self.currentIndex], \
                        self.r_layers_list[
                            self.currentIndex + 1]
                    self.layer_flush('回归器')
                    self.treeWidget.setCurrentItem(self.r_ui_list[self.currentIndex + 1])
                    print(self.treeWidget.currentItem().parent().text(0) + '->下移当前隐藏层')

    def dialog_selection(self, text):
        self.layer_selection = text

    def load_models_and_dataset(self, text):
        fname = path.dirname(path.realpath(sys.argv[0]))
        if text == 'All':
            dataset = str(fname + '\\' + 'data' + '\\' + 'PTA_DATA.xls')
            print('加载数据集:', dataset)
            data = read_excel(dataset, header=None)
            data = data.values
            self.x = torch.from_numpy(data[:, :17])
            self.y = torch.unsqueeze(torch.from_numpy(data[:, 17]), dim=1)

            print('train_data shape:', self.x.shape)
            print('train_label shape:', self.y.shape)

            print(fname + '\\' + 'g.h5')
            print(fname + '\\' + 'd.h5')
            print(fname + '\\' + 'r.h5')
        elif text == 'Generator':
            print(text)

        elif text == 'Discriminator':
            print(text)

        elif text == 'Regressor':
            print(text)

        elif text == 'Dataset':
            fname, _ = QFileDialog.getOpenFileName(self, '选择数据集文件', '.', '数据集文件(*.xls)')
            if fname == '':
                return
            print('加载数据集:', fname)
            data = read_excel(fname, header=None)
            data = data.values
            self.x = torch.from_numpy(data[:, :17])
            self.y = torch.unsqueeze(torch.from_numpy(data[:, 17]), dim=1)

            print('train_data shape:', self.x.shape)
            print('train_label shape:', self.y.shape)

    def build_model_structure(self):
        self.net = nn.Sequential()
        for i in range(len(self.g_layers_list)):
            # add layer type
            if self.g_layers_list[i].lineEdit.text() == 'Dense':
                if i == 0:
                    self.net.add_module('Input',
                                        nn.Linear(in_features=17, out_features=self.g_layers_list[i].spinBox.value(),
                                                  dtype=float))
                else:
                    self.net.add_module('Dense' + str(i),
                                        nn.Linear(in_features=self.g_layers_list[i-1].spinBox.value(),
                                                  out_features=self.g_layers_list[i].spinBox.value(),
                                                  dtype=float)
                                        )
            elif self.g_layers_list[i].lineEdit.text() == 'Conv':
                pass

            # add activation
            activation = self.g_layers_list[i].comboBox_2.currentText()
            if activation == 'relu':
                self.net.add_module(activation + str(i),
                                    nn.ReLU())
            elif activation == 'LeakyReLu':
                self.net.add_module(activation + str(i),
                                    nn.LeakyReLU())
            elif activation == 'elu':
                self.net.add_module(activation + str(i),
                                    nn.ELU())
            elif activation == 'softmax':
                self.net.add_module(activation + str(i),
                                    nn.Softmax())

        self.net.add_module('Output',
                            nn.Linear(in_features=self.g_layers_list[i].spinBox.value(), out_features=1,
                                      dtype=float))

        print(self.net)
        self.compile_model()
        self.train_model()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print(self.model().to(device))

    def compile_model(self):
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.0001)
        self.loss_function = torch.nn.MSELoss()

    def train_model(self):
        for i in range(self.epoch):
            self.app.processEvents()
            print("-------第 {} 轮训练开始-------".format(i + 1))
            # 训练步骤开始
            self.net.train()

            # 梯度清零
            self.optimizer.zero_grad()

            output = self.net(self.x)

            # 梯度反传
            loss = self.loss_function(output, self.y)
            loss.backward()
            self.optimizer.step()   # 更新所有参数

            if i % 10 == 0:
                print("epoch: {}, loss is: {}".format(i, loss.item()))


class Regressor(QWidget, UI_regressor.Ui_Form):
    def __init__(self):
        super(Regressor, self).__init__()
        self.setupUi(self)