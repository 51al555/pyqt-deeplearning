# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demo.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1398, 842)
        font = QtGui.QFont()
        font.setPointSize(15)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(0, 0, 211, 330))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setMinimumSize(QtCore.QSize(0, 0))
        self.scrollArea.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 209, 328))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents)
        self.horizontalLayout.setContentsMargins(8, 8, 8, 8)
        self.horizontalLayout.setSpacing(8)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.toolBox = QtWidgets.QToolBox(self.scrollAreaWidgetContents)
        self.toolBox.setMinimumSize(QtCore.QSize(0, 0))
        self.toolBox.setObjectName("toolBox")
        self.rCWGAN = QtWidgets.QWidget()
        self.rCWGAN.setGeometry(QtCore.QRect(0, 0, 193, 192))
        self.rCWGAN.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.rCWGAN.setObjectName("rCWGAN")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.rCWGAN)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.textBrowser = QtWidgets.QTextBrowser(self.rCWGAN)
        self.textBrowser.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.textBrowser.setObjectName("textBrowser")
        self.horizontalLayout_3.addWidget(self.textBrowser)
        self.toolBox.addItem(self.rCWGAN, "")
        self.WGAN_GP = QtWidgets.QWidget()
        self.WGAN_GP.setGeometry(QtCore.QRect(0, 0, 193, 192))
        self.WGAN_GP.setObjectName("WGAN_GP")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.WGAN_GP)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.WGAN_GP)
        self.textBrowser_2.setStyleSheet("")
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.horizontalLayout_4.addWidget(self.textBrowser_2)
        self.toolBox.addItem(self.WGAN_GP, "")
        self.infoWGAN = QtWidgets.QWidget()
        self.infoWGAN.setGeometry(QtCore.QRect(0, 0, 193, 192))
        self.infoWGAN.setObjectName("infoWGAN")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.infoWGAN)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.textBrowser_3 = QtWidgets.QTextBrowser(self.infoWGAN)
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.horizontalLayout_5.addWidget(self.textBrowser_3)
        self.toolBox.addItem(self.infoWGAN, "")
        self.horizontalLayout.addWidget(self.toolBox)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(220, 10, 1171, 621))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setMovable(True)
        self.tabWidget.setObjectName("tabWidget")
        self.model_config = QtWidgets.QWidget()
        self.model_config.setObjectName("model_config")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.model_config)
        self.verticalLayout.setContentsMargins(7, 7, 7, 7)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_3 = QtWidgets.QGroupBox(self.model_config)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.treeWidget = QtWidgets.QTreeWidget(self.groupBox_3)
        self.treeWidget.setGeometry(QtCore.QRect(10, 20, 661, 371))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.treeWidget.setFont(font)
        self.treeWidget.setObjectName("treeWidget")
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        self.layoutWidget = QtWidgets.QWidget(self.groupBox_3)
        self.layoutWidget.setGeometry(QtCore.QRect(50, 400, 471, 51))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.bn_plus = QtWidgets.QPushButton(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bn_plus.sizePolicy().hasHeightForWidth())
        self.bn_plus.setSizePolicy(sizePolicy)
        self.bn_plus.setMaximumSize(QtCore.QSize(40, 16777215))
        self.bn_plus.setText("")
        self.bn_plus.setObjectName("bn_plus")
        self.horizontalLayout_2.addWidget(self.bn_plus)
        self.bn_minus = QtWidgets.QPushButton(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bn_minus.sizePolicy().hasHeightForWidth())
        self.bn_minus.setSizePolicy(sizePolicy)
        self.bn_minus.setMaximumSize(QtCore.QSize(40, 16777215))
        self.bn_minus.setText("")
        self.bn_minus.setObjectName("bn_minus")
        self.horizontalLayout_2.addWidget(self.bn_minus)
        spacerItem = QtWidgets.QSpacerItem(225, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.bn_go_up = QtWidgets.QPushButton(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bn_go_up.sizePolicy().hasHeightForWidth())
        self.bn_go_up.setSizePolicy(sizePolicy)
        self.bn_go_up.setMaximumSize(QtCore.QSize(40, 16777215))
        self.bn_go_up.setText("")
        self.bn_go_up.setObjectName("bn_go_up")
        self.horizontalLayout_2.addWidget(self.bn_go_up)
        self.bn_go_down = QtWidgets.QPushButton(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bn_go_down.sizePolicy().hasHeightForWidth())
        self.bn_go_down.setSizePolicy(sizePolicy)
        self.bn_go_down.setMaximumSize(QtCore.QSize(40, 16777215))
        self.bn_go_down.setText("")
        self.bn_go_down.setObjectName("bn_go_down")
        self.horizontalLayout_2.addWidget(self.bn_go_down)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_4.setGeometry(QtCore.QRect(-10, 480, 971, 60))
        self.groupBox_4.setMaximumSize(QtCore.QSize(16777215, 60))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.groupBox_4)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.widget_2 = QtWidgets.QWidget(self.groupBox_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy)
        self.widget_2.setMaximumSize(QtCore.QSize(16777215, 35))
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_7.setContentsMargins(5, 0, 5, 0)
        self.horizontalLayout_7.setSpacing(0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.progressBar = QtWidgets.QProgressBar(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy)
        self.progressBar.setMaximumSize(QtCore.QSize(16777215, 30))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_7.addWidget(self.progressBar)
        self.bn_train = QtWidgets.QPushButton(self.widget_2)
        self.bn_train.setMinimumSize(QtCore.QSize(150, 0))
        self.bn_train.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.bn_train.setObjectName("bn_train")
        self.horizontalLayout_7.addWidget(self.bn_train)
        self.horizontalLayout_6.addWidget(self.widget_2)
        self.bn_load_dataset = QtWidgets.QPushButton(self.groupBox_3)
        self.bn_load_dataset.setGeometry(QtCore.QRect(690, 50, 111, 31))
        self.bn_load_dataset.setObjectName("bn_load_dataset")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_2.setGeometry(QtCore.QRect(700, 90, 411, 161))
        self.groupBox_2.setObjectName("groupBox_2")
        self.bn_load_model_g = QtWidgets.QPushButton(self.groupBox_2)
        self.bn_load_model_g.setGeometry(QtCore.QRect(0, 30, 111, 31))
        self.bn_load_model_g.setObjectName("bn_load_model_g")
        self.generator_path = QtWidgets.QLabel(self.groupBox_2)
        self.generator_path.setGeometry(QtCore.QRect(120, 20, 281, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.generator_path.setFont(font)
        self.generator_path.setText("")
        self.generator_path.setObjectName("generator_path")
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_5.setGeometry(QtCore.QRect(710, 250, 411, 141))
        self.groupBox_5.setObjectName("groupBox_5")
        self.bn_load_model_d = QtWidgets.QPushButton(self.groupBox_5)
        self.bn_load_model_d.setGeometry(QtCore.QRect(0, 20, 111, 31))
        self.bn_load_model_d.setObjectName("bn_load_model_d")
        self.discriminator_path = QtWidgets.QLabel(self.groupBox_5)
        self.discriminator_path.setGeometry(QtCore.QRect(110, 20, 281, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.discriminator_path.setFont(font)
        self.discriminator_path.setText("")
        self.discriminator_path.setObjectName("discriminator_path")
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_6.setGeometry(QtCore.QRect(730, 360, 421, 141))
        self.groupBox_6.setObjectName("groupBox_6")
        self.bn_load_model_r = QtWidgets.QPushButton(self.groupBox_6)
        self.bn_load_model_r.setGeometry(QtCore.QRect(0, 30, 111, 31))
        self.bn_load_model_r.setObjectName("bn_load_model_r")
        self.regressor_path = QtWidgets.QLabel(self.groupBox_6)
        self.regressor_path.setGeometry(QtCore.QRect(120, 20, 281, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.regressor_path.setFont(font)
        self.regressor_path.setText("")
        self.regressor_path.setObjectName("regressor_path")
        self.dataset_path = QtWidgets.QLabel(self.groupBox_3)
        self.dataset_path.setGeometry(QtCore.QRect(840, 40, 301, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.dataset_path.setFont(font)
        self.dataset_path.setText("")
        self.dataset_path.setObjectName("dataset_path")
        self.bn_load_all = QtWidgets.QPushButton(self.groupBox_3)
        self.bn_load_all.setGeometry(QtCore.QRect(690, 20, 111, 31))
        self.bn_load_all.setObjectName("bn_load_all")
        self.verticalLayout.addWidget(self.groupBox_3)
        self.tabWidget.addTab(self.model_config, "")
        self.model_output = QtWidgets.QWidget()
        self.model_output.setObjectName("model_output")
        self.tabWidget.addTab(self.model_output, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1398, 26))
        self.menuBar.setObjectName("menuBar")
        self.menubar = QtWidgets.QMenu(self.menuBar)
        self.menubar.setObjectName("menubar")
        self.menubar_2 = QtWidgets.QMenu(self.menuBar)
        self.menubar_2.setObjectName("menubar_2")
        MainWindow.setMenuBar(self.menuBar)
        self.dockWidget = QtWidgets.QDockWidget(MainWindow)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.dockWidget.setFont(font)
        self.dockWidget.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea|QtCore.Qt.TopDockWidgetArea)
        self.dockWidget.setObjectName("dockWidget")
        self.dockWidgetContents_2 = QtWidgets.QWidget()
        self.dockWidgetContents_2.setObjectName("dockWidgetContents_2")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.dockWidgetContents_2)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.textEdit = QtWidgets.QTextEdit(self.dockWidgetContents_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.textEdit.setFont(font)
        self.textEdit.setStyleSheet("")
        self.textEdit.setReadOnly(True)
        self.textEdit.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayout_8.addWidget(self.textEdit)
        self.dockWidget.setWidget(self.dockWidgetContents_2)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(8), self.dockWidget)
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setObjectName("action")
        self.action_3 = QtWidgets.QAction(MainWindow)
        self.action_3.setObjectName("action_3")
        self.menuBar.addAction(self.menubar.menuAction())
        self.menuBar.addAction(self.menubar_2.menuAction())

        self.retranslateUi(MainWindow)
        self.toolBox.setCurrentIndex(0)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "深度学习工具箱"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:15pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">WGAN-GP + CGAN + 回归器(多层感知机)</p></body></html>"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.rCWGAN), _translate("MainWindow", "rCWGAN"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.WGAN_GP), _translate("MainWindow", "WGAN-GP"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.infoWGAN), _translate("MainWindow", "infoWGAN"))
        self.groupBox_3.setTitle(_translate("MainWindow", "结构"))
        self.treeWidget.headerItem().setText(0, _translate("MainWindow", "name"))
        self.treeWidget.headerItem().setText(1, _translate("MainWindow", "layers"))
        self.treeWidget.headerItem().setText(2, _translate("MainWindow", "units"))
        self.treeWidget.headerItem().setText(3, _translate("MainWindow", "activation"))
        self.treeWidget.headerItem().setText(4, _translate("MainWindow", "kernel_initializer"))
        self.treeWidget.headerItem().setText(5, _translate("MainWindow", "others"))
        __sortingEnabled = self.treeWidget.isSortingEnabled()
        self.treeWidget.setSortingEnabled(False)
        self.treeWidget.topLevelItem(0).setText(0, _translate("MainWindow", "生成器"))
        self.treeWidget.topLevelItem(0).setText(1, _translate("MainWindow", "..."))
        self.treeWidget.topLevelItem(0).setText(2, _translate("MainWindow", "..."))
        self.treeWidget.topLevelItem(0).setText(3, _translate("MainWindow", "..."))
        self.treeWidget.topLevelItem(0).setText(4, _translate("MainWindow", "..."))
        self.treeWidget.topLevelItem(0).setText(5, _translate("MainWindow", "..."))
        self.treeWidget.topLevelItem(1).setText(0, _translate("MainWindow", "判别器"))
        self.treeWidget.topLevelItem(1).setText(1, _translate("MainWindow", "..."))
        self.treeWidget.topLevelItem(1).setText(2, _translate("MainWindow", "..."))
        self.treeWidget.topLevelItem(1).setText(3, _translate("MainWindow", "..."))
        self.treeWidget.topLevelItem(1).setText(4, _translate("MainWindow", "..."))
        self.treeWidget.topLevelItem(1).setText(5, _translate("MainWindow", "..."))
        self.treeWidget.topLevelItem(2).setText(0, _translate("MainWindow", "回归器"))
        self.treeWidget.topLevelItem(2).setText(1, _translate("MainWindow", "..."))
        self.treeWidget.topLevelItem(2).setText(2, _translate("MainWindow", "..."))
        self.treeWidget.topLevelItem(2).setText(3, _translate("MainWindow", "..."))
        self.treeWidget.topLevelItem(2).setText(4, _translate("MainWindow", "..."))
        self.treeWidget.topLevelItem(2).setText(5, _translate("MainWindow", "..."))
        self.treeWidget.setSortingEnabled(__sortingEnabled)
        self.groupBox_4.setTitle(_translate("MainWindow", "调试"))
        self.bn_train.setText(_translate("MainWindow", "训练"))
        self.bn_load_dataset.setText(_translate("MainWindow", "加载数据集"))
        self.groupBox_2.setTitle(_translate("MainWindow", "生成器"))
        self.bn_load_model_g.setText(_translate("MainWindow", "加载模型"))
        self.groupBox_5.setTitle(_translate("MainWindow", "判别器"))
        self.bn_load_model_d.setText(_translate("MainWindow", "加载模型"))
        self.groupBox_6.setTitle(_translate("MainWindow", "回归器"))
        self.bn_load_model_r.setText(_translate("MainWindow", "加载模型"))
        self.bn_load_all.setText(_translate("MainWindow", "一键加载"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.model_config), _translate("MainWindow", "模型配置"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.model_output), _translate("MainWindow", "模型输出"))
        self.menubar.setTitle(_translate("MainWindow", "终端"))
        self.menubar_2.setTitle(_translate("MainWindow", "帮助"))
        self.dockWidget.setWindowTitle(_translate("MainWindow", "终端"))
        self.action.setText(_translate("MainWindow", "显示"))
        self.action_3.setText(_translate("MainWindow", "清空内容"))
