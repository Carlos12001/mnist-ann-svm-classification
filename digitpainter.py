# Copyright (C) 2024 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2024
# Proyecto 1

import os
import tkinter as tk

# Some version magic here
from PIL import Image,ImageDraw

try:
    from PIL import ImageOps, Resampling
    resampling_filter = Resampling.LANCZOS
except ImportError:
    resampling_filter = Image.LANCZOS

import numpy as np

class DigitPainter(tk.Frame):
  # Canvas to paint digits
  
  def __init__(self,master=None):
    super().__init__(master)
    self.master = master
    self.pack()
    
    self.lastX, self.lastY = 0, 0        
    self.pencil=55        
    self.imgside=448
    
    self.h=int((self.pencil-1)/2)
    self.create_widgets()

    self.classifiers = {}

    self.buttons_state = tk.DISABLED
        
        
  def create_widgets(self):
    self.master.geometry("448x512")  # 448=28x16
    self.master.title("Drawing window")
    
    self.img = Image.new(mode="L",size=(self.imgside,self.imgside),color=(0)) # "L" means 8-bits        
    self.painter=ImageDraw.Draw(self.img)
    self.canvas= tk.Canvas(self.master,width=self.imgside,height=self.imgside,bg="black")
               
    self.canvas.bind( "<Button-1>", self.get_x_and_y )
    self.canvas.bind( "<B1-Motion>", self.draw_box )
    self.canvas.pack() #expand = YES, fill = BOTH)
        
    # Botones y acciones
    self.frame = tk.Frame(self.master)
    self.frame.pack()
    
    self.resetButton=tk.Button(self.frame,text="Reset",command= self.erase_canvas) 
    self.resetButton.pack(side= tk.LEFT)

    self.result_var = tk.StringVar()  # Variable to hold the prediction result
    self.result_label = tk.Label(self.master, textvariable=self.result_var)
    self.result_label.pack() 
        
  # Add a new classifier with the corresponding button
  def add_classifier(self,classifier):
    self.classifiers[classifier.name] = classifier
    button = tk.Button(self.frame,text=classifier.name,
                      command=lambda: self.predict(classifier))
    button.pack(side=tk.RIGHT)

      
  # Called when the button is pressed the first time
  def get_x_and_y(self,event):
    self.lastX,self.lastY = event.x, event.y

    if self.buttons_state != tk.NORMAL:
      for widget in self.frame.winfo_children():
        if isinstance(widget, tk.Button):
          widget["state"] = tk.NORMAL
      self.buttons_state = tk.NORMAL
        
  # Called on a drag event
  def draw_box(self,event):
        
    self.canvas.create_line((self.lastX,self.lastY,event.x,event.y),fill="white",width=self.pencil)
    self.canvas.create_oval((event.x-self.h,event.y-self.h,event.x+self.h,event.y+self.h),
                            fill="white",outline="white")
        
    self.painter.line([(self.lastX,self.lastY),(event.x,event.y)],fill="white",width=self.pencil)
    self.painter.ellipse([(event.x-self.h,event.y-self.h),(event.x+self.h,event.y+self.h)],
                         fill="white",outline="white")        
        
    self.lastX,self.lastY=event.x,event.y

        
  #Encargado de borrar todo (para bóton)
  def erase_canvas(self):
    
    self.canvas.delete("all")  

    if self.buttons_state != tk.DISABLED:
      for widget in self.frame.winfo_children():
        if isinstance(widget, tk.Button):
          widget["state"] = tk.DISABLED
      self.buttons_state = tk.DISABLED
        
    self.painter.rectangle([0,0,self.imgside,self.imgside],fill="black",outline="black")
        
  def getMNISTImage(self):
    return np.array(self.img.resize((28,28),resampling_filter))
        
  def predict(self,classifier):
    img = self.getMNISTImage()
    pred_digit=classifier.predict(img)
    if pred_digit is not None:
        self.result_var.set(f"Predicted digit: {pred_digit}")
        print(f"Predicted digit: {pred_digit}")
    else:
        self.result_var.set("Prediction failed")
