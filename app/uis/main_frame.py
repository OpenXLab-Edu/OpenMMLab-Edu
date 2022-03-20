from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtGui

from uis.main_windows import Ui_Form


class MMEdu_Frame(QMainWindow):
    def __init__(self):
        super(MMEdu_Frame,self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)