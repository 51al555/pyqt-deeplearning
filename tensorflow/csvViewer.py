from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import csv
import os


class TableView(QMainWindow):
    def __init__(self, dataset='test.csv'):
        super(TableView, self).__init__()
        self.dataset = dataset
        self.setWindowTitle(dataset)
        self.resize(1600, 900)
        # self.setWindowIcon(QIcon('./images/icon.jpg'))

        # 菜单栏
        bar = self.menuBar()
        # 添加选项
        file = bar.addMenu('文件')
        edit = bar.addMenu('查找')
        # 添加“新建”的菜单项-?“保存”
        save = QAction('保存', self)
        save.setShortcut('ctrl+s')
        file.addAction(save)
        exfind = QAction('精确查找', self)
        edit.addAction(exfind)

        save.triggered.connect(self.prosess)
        edit.triggered.connect(self.exfind)

        # 状态栏
        self.sb = QStatusBar()
        self.setStatusBar(self.sb)

        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['Sign', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5',
                                              'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13',
                                              'Z0', 'Z1', 'Z2'])  # 字段

        self.tableview = QTableView()
        self.tableview.resizeRowsToContents()
        self.tableview.resizeColumnsToContents()
        # self.tableview.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 自动扩展列宽
        # self.tableview.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 自动扩展列宽
        self.tableview.setModel(self.model)  # 关联模型

        with open(self.dataset) as f:
            reader = csv.reader(f)
            self.hang = 0
            for row in reader:
                for i, val in enumerate(row):
                    self.model.setItem(self.hang, i, QStandardItem(str(val)))
                self.hang += 1

        self.setCentralWidget(self.tableview)  # 主框架充满整个窗口

        self.row = self.model.rowCount()
        self.column = self.model.columnCount()

    def exfind(self):
        QMessageBox.information(self, '先画个饼', '该功能待开发中...')

    def prosess(self, a):
        # print(self.sender().text())
        with open('cache.csv', 'a', newline="") as f1:
            f_csv = csv.writer(f1)
            for i in range(self.row):
                datalist = []
                for j in range(self.column):
                    datalist.append(self.model.data(self.model.index(i, j)))
                f_csv.writerow(datalist)
        self.sb.showMessage('数据已保存', 1000)
        try:
            os.remove(self.dataset)
            os.rename('cache.csv', self.dataset)
        except:
            os.rename('cache.csv', self.dataset)


if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建QApplication类的实例(应用程序)
    main = TableView()
    main.show()

    sys.exit(app.exec_())