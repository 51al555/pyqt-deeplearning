# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI_rcwgan.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1181, 765)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox_3 = QtWidgets.QGroupBox(Form)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.bn_load_all = QtWidgets.QPushButton(self.groupBox_3)
        self.bn_load_all.setObjectName("bn_load_all")
        self.horizontalLayout_3.addWidget(self.bn_load_all)
        self.bn_load_dataset = QtWidgets.QPushButton(self.groupBox_3)
        self.bn_load_dataset.setObjectName("bn_load_dataset")
        self.horizontalLayout_3.addWidget(self.bn_load_dataset)
        self.bn_check_dataset = QtWidgets.QPushButton(self.groupBox_3)
        self.bn_check_dataset.setObjectName("bn_check_dataset")
        self.horizontalLayout_3.addWidget(self.bn_check_dataset)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.bn_load_model_g = QtWidgets.QPushButton(self.groupBox_2)
        self.bn_load_model_g.setObjectName("bn_load_model_g")
        self.horizontalLayout_10.addWidget(self.bn_load_model_g)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_5.setObjectName("groupBox_5")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.groupBox_5)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.bn_load_model_d = QtWidgets.QPushButton(self.groupBox_5)
        self.bn_load_model_d.setObjectName("bn_load_model_d")
        self.horizontalLayout_11.addWidget(self.bn_load_model_d)
        self.verticalLayout.addWidget(self.groupBox_5)
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_6.setObjectName("groupBox_6")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_6)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.bn_load_model_r = QtWidgets.QPushButton(self.groupBox_6)
        self.bn_load_model_r.setObjectName("bn_load_model_r")
        self.verticalLayout_3.addWidget(self.bn_load_model_r)
        self.bn_predict = QtWidgets.QPushButton(self.groupBox_6)
        self.bn_predict.setEnabled(False)
        self.bn_predict.setObjectName("bn_predict")
        self.verticalLayout_3.addWidget(self.bn_predict)
        self.verticalLayout.addWidget(self.groupBox_6)
        self.gridLayout.addLayout(self.verticalLayout, 0, 1, 1, 1)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_4.setMaximumSize(QtCore.QSize(16777215, 60))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.progressBar = QtWidgets.QProgressBar(self.groupBox_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy)
        self.progressBar.setMaximumSize(QtCore.QSize(16777215, 30))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout_2.addWidget(self.progressBar, 0, 0, 1, 1)
        self.bn_train = QtWidgets.QPushButton(self.groupBox_4)
        self.bn_train.setMinimumSize(QtCore.QSize(0, 0))
        self.bn_train.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.bn_train.setObjectName("bn_train")
        self.gridLayout_2.addWidget(self.bn_train, 0, 1, 1, 1)
        self.bn_continue_train = QtWidgets.QPushButton(self.groupBox_4)
        self.bn_continue_train.setEnabled(False)
        self.bn_continue_train.setObjectName("bn_continue_train")
        self.gridLayout_2.addWidget(self.bn_continue_train, 0, 2, 1, 1)
        self.bn_saveWeight = QtWidgets.QPushButton(self.groupBox_4)
        self.bn_saveWeight.setEnabled(False)
        self.bn_saveWeight.setObjectName("bn_saveWeight")
        self.gridLayout_2.addWidget(self.bn_saveWeight, 0, 3, 1, 1)
        self.gridLayout_2.setColumnStretch(0, 5)
        self.gridLayout_2.setColumnStretch(1, 1)
        self.gridLayout_2.setColumnStretch(2, 1)
        self.gridLayout.addWidget(self.groupBox_4, 1, 0, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox_3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox_3.setTitle(_translate("Form", "结构"))
        self.bn_load_all.setText(_translate("Form", "一键加载"))
        self.bn_load_dataset.setText(_translate("Form", "加载数据集"))
        self.bn_check_dataset.setText(_translate("Form", "查看数据集"))
        self.groupBox_2.setTitle(_translate("Form", "生成器"))
        self.bn_load_model_g.setText(_translate("Form", "加载模型"))
        self.groupBox_5.setTitle(_translate("Form", "判别器"))
        self.bn_load_model_d.setText(_translate("Form", "加载模型"))
        self.groupBox_6.setTitle(_translate("Form", "回归器"))
        self.bn_load_model_r.setText(_translate("Form", "加载模型"))
        self.bn_predict.setText(_translate("Form", "预测"))
        self.groupBox_4.setTitle(_translate("Form", "调试"))
        self.bn_train.setText(_translate("Form", "从头训练"))
        self.bn_continue_train.setText(_translate("Form", "继续训练"))
        self.bn_saveWeight.setText(_translate("Form", "保存权重"))

