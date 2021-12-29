import os
import sys
from functools import partial

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTreeWidgetItem, QHeaderView, QDialog, QAction, \
    QMessageBox
from demo_module import Ui_MainWindow
import rcwgan
import Dense
import Choose_Layer_Type


# QTreeWidget在调用各类删除方法时，会同时调用QTreeWidgetItem的析构（可能也会调用QWidget的析构），导致内存删除，需要深拷贝到新对象


class Layers_Choose(QDialog, Choose_Layer_Type.Ui_Dialog):
    def __init__(self):
        super(Layers_Choose, self).__init__()
        self.setupUi(self)
        layers = ['Dense', 'Dropout', 'Conv']
        self.comboBox.addItems(layers)


class Layer_Dense(QWidget, Dense.Ui_Form):
    def __init__(self):
        super(Layer_Dense, self).__init__()
        self.setupUi(self)
        activation = ['linear', 'elu', 'relu', 'LeakyReLu']
        self.comboBox_2.addItems(activation)

    # 自定义深拷贝
    def copy(self, old):
        self.spinBox.setValue(old.spinBox.value())
        self.comboBox_2.setCurrentIndex(old.comboBox_2.currentIndex())


class rCWGAN(QWidget, rcwgan.Ui_Form):
    layer_selection = 'Dense'

    g_ui_list = []
    g_layers_list = []

    d_ui_list = []
    d_layers_list = []

    r_ui_list = []
    r_layers_list = []

    currentIndex = 0

    def __init__(self):
        super(rCWGAN, self).__init__()
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
        self.bn_load_all.clicked.connect(partial(self.load_models, 'All'))
        self.bn_load_model_g.clicked.connect(partial(self.load_models, 'Generator'))
        self.bn_load_model_d.clicked.connect(partial(self.load_models, 'Discriminator'))
        self.bn_load_model_r.clicked.connect(partial(self.load_models, 'Regressor'))

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

    def load_models(self, text):
        fname = os.path.dirname(os.path.realpath(sys.argv[0]))
        if text == 'All':
            print(fname + '\\' + 'PTA_DATA.xls')
            print(fname + '\\' + 'g.h5')
            print(fname + '\\' + 'd.h5')
            print(fname + '\\' + 'r.h5')
        elif text == 'Generator':
            print(text)

        elif text == 'Discriminator':
            print(text)

        elif text == 'Regressor':
            print(text)


class EmittingStr(QObject):
    textWritten = pyqtSignal(str)  # 定义一个发送str的信号

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class PyQt5_And_TensorFlow_Demo(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(PyQt5_And_TensorFlow_Demo, self).__init__()
        self.setupUi(self)

        sys.stdout = EmittingStr(textWritten=self.outputWritten)
        sys.stderr = EmittingStr(textWritten=self.outputWritten)

        self.iniUI()
        self.show()

        self.model = rCWGAN()
        self.model.show()

    def iniUI(self):
        # menu
        terminal_show = QAction('显示', self)
        terminal_clear = QAction('清空', self)
        terminal_clear.setShortcut('ctrl+z')
        self.menubar.addAction(terminal_show)
        self.menubar.addSeparator()
        self.menubar.addAction(terminal_clear)

        info = QAction('关于训练', self)
        self.menubar_2.addAction(info)

        # slots and signals
        terminal_show.triggered.connect(self.menu_clicked)
        terminal_clear.triggered.connect(self.menu_clicked)
        info.triggered.connect(self.info_clicked)

    def menu_clicked(self):
        if self.sender().text() == '显示':
            self.dockWidget.show()
        elif self.sender().text() == '清空':
            self.textEdit.clear()

    def info_clicked(self):
        if self.sender().text() == '关于训练':
            QMessageBox.information(self, '关于训练', '1、已加载模型的网络默认继续训练，未加载模型的网络默认按照'
                                                  '自定义结构从零开始训练。\n\n2、选择“继续训练”，未改变结构的网络在'
                                                  '上一次基础上继续进行训练，改变结构的网络从起始状态开始训练。')

    def outputWritten(self, text):
        self.textEdit.moveCursor(self.textEdit.textCursor().End)
        self.textEdit.insertPlainText(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = PyQt5_And_TensorFlow_Demo()
    sys.exit(app.exec_())
