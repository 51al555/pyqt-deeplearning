from functools import partial

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QWidget, QHeaderView, QTreeWidgetItem

import UI_Choose_Layer_Type
import UI_Dense
import UI_Model_Structure
import csvViewer

dataset_viewer = None


def check_dataset(dataset='./data/PTA_DATA.csv'):
    global dataset_viewer
    dataset_viewer = csvViewer.TableView(dataset)
    dataset_viewer.show()


class Model_Structure(QWidget, UI_Model_Structure.Ui_Structure):
    layer_selection = 'Dense'

    ui_list = []
    layers_list = []
    currentIndex = 0

    def __init__(self):
        super(Model_Structure, self).__init__()
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

    def layer_add(self):
        dialog = Layers_Choose()
        self.layer_selection = 'Dense'
        dialog.comboBox.currentTextChanged.connect(self.dialog_selection)
        res = dialog.exec_()
        if res == QDialog.Accepted:
            if self.layer_selection == 'Dense':
                if self.treeWidget.currentItem().parent() is not None:
                    self.currentIndex = self.treeWidget.currentIndex().row() + 1

                    self.layers_list.insert(self.currentIndex, Layer_Dense())

                    # UI
                    self.ui_list.insert(self.currentIndex, QTreeWidgetItem())
                    self.treeWidget.topLevelItem(0).insertChild(self.currentIndex,
                                                                self.ui_list[self.currentIndex])
                    self.treeWidget.setItemWidget(self.ui_list[self.currentIndex], 1,
                                                  self.layers_list[self.currentIndex])
                    print(self.treeWidget.currentItem().parent().text(0) + '->添加隐藏层')
                else:
                    self.layers_list.append(Layer_Dense())

                    # UI
                    self.ui_list.append(QTreeWidgetItem())
                    self.treeWidget.topLevelItem(0).addChild(self.ui_list[-1])
                    self.treeWidget.setItemWidget(self.ui_list[-1], 1, self.layers_list[-1])
                    print(self.treeWidget.currentItem().text(0) + '->添加隐藏层')

            elif self.layer_selection == 'Dropout':
                print(self.layer_selection)
            elif self.layer_selection == 'Conv':
                print(self.layer_selection)

    def layer_delete(self):
        if self.treeWidget.currentItem().parent() is not None:
            self.currentIndex = self.treeWidget.currentIndex().row()

            self.layers_list.pop(self.currentIndex)

            # UI
            self.treeWidget.topLevelItem(0).removeChild(self.ui_list[self.currentIndex])
            self.ui_list.pop(self.currentIndex)
            print(self.treeWidget.currentItem().parent().text(0) + '->删除隐藏层')
        else:
            if len(self.layers_list) != 0:
                self.layers_list.pop()

                # UI
                self.treeWidget.topLevelItem(0).removeChild(self.ui_list[-1])
                self.ui_list.pop()
                print(self.treeWidget.currentItem().text(0) + '->删除隐藏层')

    def layer_flush(self, txt):
        layer_list = []
        for i in range(len(self.layers_list)):
            layer_list.append(Layer_Dense())
            layer_list[i].copy(self.layers_list[i])

        self.treeWidget.topLevelItem(0).takeChildren()
        for i in range(len(self.layers_list)):
            self.treeWidget.topLevelItem(0).addChild(self.ui_list[i])
            self.treeWidget.setItemWidget(self.ui_list[i], 1, layer_list[i])

        self.layers_list = layer_list[:]

    def layer_go_up(self):
        if self.treeWidget.currentItem().parent() is not None:
            self.currentIndex = self.treeWidget.currentIndex().row()
            if self.currentIndex != 0:
                self.layers_list[self.currentIndex - 1], self.layers_list[self.currentIndex] = \
                    self.layers_list[
                        self.currentIndex], \
                    self.layers_list[
                        self.currentIndex - 1]
                self.layer_flush('回归器')
                self.treeWidget.setCurrentItem(self.ui_list[self.currentIndex - 1])
                print(self.treeWidget.currentItem().parent().text(0) + '->上移当前隐藏层')

    def layer_go_down(self):
        if self.treeWidget.currentItem().parent() is not None:
            self.currentIndex = self.treeWidget.currentIndex().row()
            if self.currentIndex != (len(self.layers_list) - 1):
                self.layers_list[self.currentIndex + 1], self.layers_list[self.currentIndex] = \
                    self.layers_list[
                        self.currentIndex], \
                    self.layers_list[
                        self.currentIndex + 1]
                self.layer_flush('回归器')
                self.treeWidget.setCurrentItem(self.ui_list[self.currentIndex + 1])
                print(self.treeWidget.currentItem().parent().text(0) + '->下移当前隐藏层')

    def dialog_selection(self, text):
        self.layer_selection = text


class Layers_Choose(QDialog, UI_Choose_Layer_Type.Ui_Dialog):
    def __init__(self):
        super(Layers_Choose, self).__init__()
        self.setupUi(self)
        layers = ['Dense', 'Dropout', 'Conv']
        self.comboBox.addItems(layers)


class Layer_Dense(QWidget, UI_Dense.Ui_Form):
    def __init__(self):
        super(Layer_Dense, self).__init__()
        self.setupUi(self)
        activation = ['None', 'elu', 'relu', 'LeakyReLu', 'softmax']
        self.comboBox_2.addItems(activation)
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(999)

    # 自定义深拷贝
    def copy(self, old):
        self.spinBox.setValue(old.spinBox.value())
        self.comboBox_2.setCurrentIndex(old.comboBox_2.currentIndex())


class Layer_Conv(QWidget):
    def __init__(self):
        super(Layer_Conv, self).__init__()

    def copy(self):
        pass