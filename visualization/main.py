from uis.main_app import MainAPP
import sys

app = MainAPP()
status = app.exec_()
sys.exit(status)

