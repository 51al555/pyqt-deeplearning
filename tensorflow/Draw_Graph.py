import sys

from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QWidget, QApplication
import UI_draw_graph

from pyqtgraph import GraphicsLayoutWidget, mkPen, InfiniteLine, TextItem


class Draw_graph(QWidget, UI_draw_graph.Ui_Form):
    def __init__(self, model):
        super(Draw_graph, self).__init__()
        self.model = model
        self.is_vline = False
        self.vline_axis = 0
        self.f_range = 100

        self.setupUi(self)

        self.initUI()

    def initUI(self):

        self.train_graph_build()

        # slots and signals
        self.comboBox.currentTextChanged.connect(self.change_graph)
        self.checkBox.stateChanged.connect(self.show_grid)
        self.checkBox_2.stateChanged.connect(self.show_vline)
        self.checkBox_3.stateChanged.connect(self.curve_follow)

        intValidator = QIntValidator()
        intValidator.setRange(1, 9999)
        self.lineEdit.setText('100')
        self.lineEdit.setValidator(intValidator)
        self.lineEdit.textChanged.connect(self.follow_range)

        self.horizontalLayout_2.addWidget(self.graph_widget)

    def train_graph_build(self):
        self.graph_widget = GraphicsLayoutWidget()
        self.train_graph = self.graph_widget.addPlot(title="Train Curve")
        self.test_graph = self.graph_widget.addPlot(title="Test Curve")

        self.curve = self.train_graph.plot(self.model.train_loss, pen=mkPen(color='r', width=2),
                                           name='train_loss')
        self.curve2 = self.test_graph.plot(self.model.test_loss, pen=mkPen(color='g', width=2),
                                           name='test_loss')

        self.train_graph.setLabel('left', 'MSE Loss')
        self.train_graph.setLabel('bottom', 'epoch')
        self.test_graph.setLabel('left', 'MSE Loss')
        self.test_graph.setLabel('bottom', 'epoch/10')

        self.train_graph.scene().sigMouseMoved.connect(self.MouseMoved)

    def predict_graph_build(self):
        self.graph_widget_2 = GraphicsLayoutWidget()
        self.pred_real_graph = self.graph_widget_2.addPlot(title="Predict Curve")
        self.pred_real_graph.addLegend()
        self.curve3 = self.pred_real_graph.plot(self.model.predict, pen=mkPen(color='b', width=2),
                                                name='Predict_Data')
        self.curve4 = self.pred_real_graph.plot(self.model.y_data, pen=mkPen(color='g', width=2),
                                                name='Real_Data')
        self.pred_real_graph.setLabel('left', 'Value')
        self.pred_real_graph.setLabel('bottom', 'epoch')

        self.pred_real_graph.scene().sigMouseMoved.connect(self.MouseMoved_pre_real)

    def show_vline(self):
        if self.checkBox_2.isChecked():
            if self.comboBox.currentText() == 'Loss曲线':
                self.train_vLine = InfiniteLine(angle=90)
                self.test_vLine = InfiniteLine(angle=90)

                self.train_graph.addItem(self.train_vLine, ignoreBounds=True)
                self.test_graph.addItem(self.test_vLine, ignoreBounds=True)

                self.train_label = TextItem()
                self.test_label = TextItem()
                self.train_graph.addItem(self.train_label, ignoreBounds=True)
                self.test_graph.addItem(self.test_label, ignoreBounds=True)

            else:
                self.pred_real_vline = InfiniteLine(angle=90)
                self.pred_real_graph.addItem(self.pred_real_vline, ignoreBounds=True)

                self.real_label = TextItem()
                self.pred_label = TextItem()
                self.pred_real_graph.addItem(self.real_label, ignoreBounds=True)
                self.pred_real_graph.addItem(self.pred_label, ignoreBounds=True)

            self.is_vline = True
        else:
            if self.is_vline:
                if self.comboBox.currentText() == 'Loss曲线':
                    self.train_graph.removeItem(self.train_vLine)
                    self.test_graph.removeItem(self.test_vLine)
                    self.train_graph.removeItem(self.train_label)
                    self.test_graph.removeItem(self.test_label)
                else:
                    self.pred_real_graph.removeItem(self.pred_real_vline)
                    self.pred_real_graph.removeItem(self.real_label)
                    self.pred_real_graph.removeItem(self.pred_label)

    def show_grid(self):
        if self.checkBox.isChecked():
            if self.comboBox.currentText() == 'Loss曲线':
                self.train_graph.showGrid(x=True, y=True)
                self.test_graph.showGrid(x=True, y=True)
            else:
                self.pred_real_graph.showGrid(x=True, y=True)
        else:
            if self.comboBox.currentText() == 'Loss曲线':
                self.train_graph.showGrid(x=False, y=False)
                self.test_graph.showGrid(x=False, y=False)
            else:
                self.pred_real_graph.showGrid(x=False, y=False)

    def change_graph(self, txt):
        if txt == '预测曲线':
            self.model.is_Loss_curves = False

            self.predict_graph_build()
            self.show_grid()
            self.is_vline = False
            self.show_vline()

            self.horizontalLayout_2.itemAt(0).widget().deleteLater()
            self.horizontalLayout_2.addWidget(self.graph_widget_2)

        elif txt == 'Loss曲线':
            self.model.is_Loss_curves = True

            self.train_graph_build()
            self.show_grid()
            self.is_vline = False
            self.show_vline()

            self.horizontalLayout_2.itemAt(0).widget().deleteLater()
            self.horizontalLayout_2.addWidget(self.graph_widget)

    def MouseMoved(self, evt):
        if self.checkBox_2.isChecked():
            train_x = self.train_graph.vb.mapSceneToView(QPointF(evt.x(), evt.y())).x()
            test_x = self.test_graph.vb.mapSceneToView(QPointF(evt.x(), evt.y())).x()
            train_y = self.train_graph.vb.mapSceneToView(QPointF(evt.x(), evt.y())).y()
            test_y = self.test_graph.vb.mapSceneToView(QPointF(evt.x(), evt.y())).y()
            self.train_vLine.setPos(train_x)
            self.test_vLine.setPos(test_x)

            train_x = int(train_x)
            test_x = int(test_x)

            if 0 < train_x < len(self.model.train_loss):
                self.train_label.setHtml(
                    "<p style='color:red;font-size:20px;'>MSE Loss：<span style='color:white;font-size:20px;'>{0}</span></p>".format(
                        self.model.train_loss[train_x - 1]))
                self.train_label.setPos(train_x, train_y)

            if 0 < test_x < len(self.model.test_loss):
                self.test_label.setHtml(
                    "<p style='color:green;font-size:20px;'>MSE Loss：<span style='color:white;font-size:20px;'>{0}</span></p>".format(
                        self.model.test_loss[test_x - 1]))
                self.test_label.setPos(test_x, test_y)

    def MouseMoved_pre_real(self, evt):
        if self.checkBox_2.isChecked():
            pre_real_x = self.pred_real_graph.vb.mapSceneToView(QPointF(evt.x(), evt.y())).x()
            pre_real_y = self.pred_real_graph.vb.mapSceneToView(QPointF(evt.x(), evt.y())).y()

            self.pred_real_vline.setPos(pre_real_x)

            pre_real_x = int(pre_real_x)

            if 0 < pre_real_x < len(self.model.y_data):
                if len(self.model.predict) == 1:
                    self.real_label.setHtml(
                        "<p style='color:green;font-size:20px;'>Real Data：<span style='color:white;font-size:20px;'>{0}</span></p><p style='color:blue;font-size:20px;'>Predict Data：<span style='color:white;font-size:20px;'>{1}</span></p>".format(
                            self.model.y_data[pre_real_x - 1], 0))
                    self.real_label.setPos(pre_real_x, pre_real_y)
                else:
                    self.real_label.setHtml(
                        "<p style='color:green;font-size:20px;'>Real Data：<span style='color:white;font-size:20px;'>{0}</span></p><p style='color:blue;font-size:20px;'>Predict Data：<span style='color:white;font-size:20px;'>{1}</span></p>".format(
                            self.model.y_data[pre_real_x - 1], self.model.predict[pre_real_x - 1]))
                    self.real_label.setPos(pre_real_x, pre_real_y)

    def curve_follow(self):
        if self.checkBox_3.isChecked():
            self.model.is_Curve_follow = True
        else:
            self.model.is_Curve_follow = False

    def follow_range(self):
        self.f_range = int(self.lineEdit.text())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = Draw_graph()
    demo.show()
    sys.exit(app.exec_())
