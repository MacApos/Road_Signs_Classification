from tkinter import *
root = Tk()

e = Entry(root, width=50, bg='red', fg='white')
e.pack()


def click():
    var = 'Hello ' + e.get()
    label=Label(root, text=var)
    label.pack()


myButton = Button(root, text='Enter your name', command=click)
myButton.pack()

root.mainloop()
