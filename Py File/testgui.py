from tkinter import *
from tkinter.filedialog import askopenfile, askopenfilename
import os

def opfile():
    #btn.fileName = askopenfilename(filetypes={("how code files", "*.hc"), ("all files","*.*")})
    os.system("explorer")
	

root = Tk()
root.geometry("100x100")
btn = Button(root, text="Hello", command=opfile())
btn.grid(row = 0, column = 3)
root.mainloop()
