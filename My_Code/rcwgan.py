# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'rcwgan.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1181, 520)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox_3 = QtWidgets.QGroupBox(Form)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.splitter = QtWidgets.QSplitter(self.groupBox_3)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.treeWidget = QtWidgets.QTreeWidget(self.layoutWidget)
        self.treeWidget.setObjectName("treeWidget")
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        self.verticalLayout_2.addWidget(self.treeWidget)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
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
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.layoutWidget_2 = QtWidgets.QWidget(self.splitter)
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.bn_load_all = QtWidgets.QPushButton(self.layoutWidget_2)
        self.bn_load_all.setObjectName("bn_load_all")
        self.horizontalLayout_9.addWidget(self.bn_load_all)
        self.bn_load_dataset = QtWidgets.QPushButton(self.layoutWidget_2)
        self.bn_load_dataset.setObjectName("bn_load_dataset")
        self.horizontalLayout_9.addWidget(self.bn_load_dataset)
        self.dataset_path = QtWidgets.QLabel(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.dataset_path.setFont(font)
        self.dataset_path.setText("")
        self.dataset_path.setObjectName("dataset_path")
        self.horizontalLayout_9.addWidget(self.dataset_path)
        self.verticalLayout_3.addLayout(self.horizontalLayout_9)
        self.groupBox_2 = QtWidgets.QGroupBox(self.layoutWidget_2)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.bn_load_model_g = QtWidgets.QPushButton(self.groupBox_2)
        self.bn_load_model_g.setObjectName("bn_load_model_g")
        self.horizontalLayout_10.addWidget(self.bn_load_model_g)
        self.generator_path = QtWidgets.QLabel(self.groupBox_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.generator_path.setFont(font)
        self.generator_path.setText("")
        self.generator_path.setObjectName("generator_path")
        self.horizontalLayout_10.addWidget(self.generator_path)
        self.verticalLayout_3.addWidget(self.groupBox_2)
        self.groupBox_5 = QtWidgets.QGroupBox(self.layoutWidget_2)
        self.groupBox_5.setObjectName("groupBox_5")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.groupBox_5)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.bn_load_model_d = QtWidgets.QPushButton(self.groupBox_5)
        self.bn_load_model_d.setObjectName("bn_load_model_d")
        self.horizontalLayout_11.addWidget(self.bn_load_model_d)
        self.discriminator_path = QtWidgets.QLabel(self.groupBox_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.discriminator_path.setFont(font)
        self.discriminator_path.setText("")
        self.discriminator_path.setObjectName("discriminator_path")
        self.horizontalLayout_11.addWidget(self.discriminator_path)
        self.verticalLayout_3.addWidget(self.groupBox_5)
        self.groupBox_6 = QtWidgets.QGroupBox(self.layoutWidget_2)
        self.groupBox_6.setObjectName("groupBox_6")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.groupBox_6)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.bn_load_model_r = QtWidgets.QPushButton(self.groupBox_6)
        self.bn_load_model_r.setObjectName("bn_load_model_r")
        self.horizontalLayout_14.addWidget(self.bn_load_model_r)
        self.regressor_path = QtWidgets.QLabel(self.groupBox_6)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.regressor_path.setFont(font)
        self.regressor_path.setText("")
        self.regressor_path.setObjectName("regressor_path")
        self.horizontalLayout_14.addWidget(self.regressor_path)
        self.verticalLayout_3.addWidget(self.groupBox_6)
        self.verticalLayout_4.addWidget(self.splitter)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_3)
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
        self.verticalLayout_4.addWidget(self.groupBox_4)
        self.horizontalLayout.addWidget(self.groupBox_3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox_3.setTitle(_translate("Form", "结构"))
        self.treeWidget.headerItem().setText(0, _translate("Form", "网络"))
        self.treeWidget.headerItem().setText(1, _translate("Form", "参数"))
        self.treeWidget.headerItem().setText(2, _translate("Form", "其他"))
        __sortingEnabled = self.treeWidget.isSortingEnabled()
        self.treeWidget.setSortingEnabled(False)
        self.treeWidget.topLevelItem(0).setText(0, _translate("Form", "生成器"))
        self.treeWidget.topLevelItem(0).setText(1, _translate("Form", "---"))
        self.treeWidget.topLevelItem(1).setText(0, _translate("Form", "判别器"))
        self.treeWidget.topLevelItem(1).setText(1, _translate("Form", "---"))
        self.treeWidget.topLevelItem(2).setText(0, _translate("Form", "回归器"))
        self.treeWidget.topLevelItem(2).setText(1, _translate("Form", "---"))
        self.treeWidget.setSortingEnabled(__sortingEnabled)
        self.bn_load_all.setText(_translate("Form", "一键加载"))
        self.bn_load_dataset.setText(_translate("Form", "加载数据集"))
        self.groupBox_2.setTitle(_translate("Form", "生成器"))
        self.bn_load_model_g.setText(_translate("Form", "加载模型"))
        self.groupBox_5.setTitle(_translate("Form", "判别器"))
        self.bn_load_model_d.setText(_translate("Form", "加载模型"))
        self.groupBox_6.setTitle(_translate("Form", "回归器"))
        self.bn_load_model_r.setText(_translate("Form", "加载模型"))
        self.groupBox_4.setTitle(_translate("Form", "调试"))
        self.bn_train.setText(_translate("Form", "训练"))

