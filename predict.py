# Copyright (C) 2024 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2024
# Proyecto 1

from digitpainter import DigitPainter
from softmax import Softmax
import tkinter as tk
from svm import SVM
from ann import ANN

root = tk.Tk()
painter = DigitPainter(master=root)

painter.add_classifier(Softmax("mnist_softmax.pt"))
painter.add_classifier(SVM("mnist_svm.pkl"))

painter.add_classifier(ANN(filename="mnist_ann.pt"))


# Add your own classifiers here (and remember the imports)


try:
    painter.mainloop()
    
except KeyboardInterrupt:
    print("\nGracefully shutting down from Ctrl+C")
finally:
    # Any cleanup code here
    print("\nCleanup complete")
