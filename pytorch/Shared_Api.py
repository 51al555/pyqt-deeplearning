from PyQt5.QtWidgets import QDialog, QWidget
import UI_Choose_Layer_Type
import UI_Dense
import csvViewer

dataset_viewer = None


def check_dataset():
    global dataset_viewer
    dataset_viewer = csvViewer.TableView(dataset='./data/PTA_DATA.csv')
    dataset_viewer.show()


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

    # 自定义深拷贝
    def copy(self, old):
        self.spinBox.setValue(old.spinBox.value())
        self.comboBox_2.setCurrentIndex(old.comboBox_2.currentIndex())


class Layer_Conv(QWidget):
    def __init__(self):
        super(Layer_Conv, self).__init__()

    def copy(self):
        pass