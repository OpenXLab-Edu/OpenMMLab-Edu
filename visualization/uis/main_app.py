from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication
from uis.main_frame import MMEdu_Frame


class MainAPP(QApplication):
    def __init__(self):
        super(MainAPP,self).__init__([])
        # app主窗口
        self.main_Window = MMEdu_Frame()
        self.main_Window.show()