import sys
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QMessageBox
import UI_demo_module
import My_Networks

# QTreeWidget在调用各类删除方法时，会同时调用QTreeWidgetItem的析构（可能也会调用QWidget的析构），导致内存删除，需要深拷贝到新对象


class EmittingStr(QObject):
    textWritten = pyqtSignal(str)  # 定义一个发送str的信号

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class PyQt5_And_Pytorch_Demo(QMainWindow, UI_demo_module.Ui_MainWindow):
    def __init__(self):
        super(PyQt5_And_Pytorch_Demo, self).__init__()
        self.setupUi(self)

        sys.stdout = EmittingStr(textWritten=self.outputWritten)
        sys.stderr = EmittingStr(textWritten=self.outputWritten)

        self.iniUI()
        self.show()

        self.model = My_Networks.rCWGAN(app)
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
        self.toolBox.currentChanged.connect(self.model_select)

    def model_select(self, index):
        self.model.close()
        if index == 0:
            self.model = My_Networks.rCWGAN(app)
            self.model.show()
        elif index == 3:
            self.model = My_Networks.Regressor()
            self.model.show()

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
    demo = PyQt5_And_Pytorch_Demo()
    sys.exit(app.exec_())
