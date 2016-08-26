from tkinter import *
import tkinter
from LSTM import *

set_path = "/home/adminz/repo/mgr/sets/"
save_results_path = "/home/adminz/results"
set_name = "close"


#####################################################################################

main = tkinter.Tk()
main.geometry('{}x{}'.format(600,600))
main.resizable(width=False, height=False)

#####################################################################################

frame1=Frame(main)

label5 = tkinter.Label(frame1,text="set path")
label5.pack(side=TOP, anchor=W, fill=X)
textSetPath = tkinter.Entry(frame1)
textSetPath.pack(side=TOP, anchor=W, fill=X)

label6 = tkinter.Label(frame1,text="path to save results")
label6.pack(side=TOP, anchor=W, fill=X)
textSaveResults = tkinter.Entry(frame1)
textSaveResults.pack(side=TOP, anchor=W, fill=X)

label7 = tkinter.Label(frame1,text="set name")
label7.pack(side=TOP, anchor=W, fill=X)
textSetName = tkinter.Entry(frame1)
textSetName.pack(side=TOP, anchor=W, fill=X)

label1=tkinter.Label(frame1,text="Iterations")
label1.pack(side=TOP, anchor=W, fill=X)
textIterations = tkinter.Entry(frame1)
textIterations.pack(side=TOP, anchor=W, fill=X)

label2 = tkinter.Label(frame1,text="Epochs")
label2.pack(side=TOP, anchor=W, fill=X)
textEpochs = tkinter.Entry(frame1)
textEpochs.pack(side=TOP, anchor=W, fill=X)

label3 = tkinter.Label(frame1,text="Alpha")
label3.pack(side=TOP, anchor=W, fill=X)
textAlpha = tkinter.Entry(frame1)
textAlpha.pack(side=TOP, anchor=W, fill=X)

label4 = tkinter.Label(frame1,text="Hidden Neurons")
label4.pack(side=TOP, anchor=W, fill=X)
textHiddenNeurons = tkinter.Entry(frame1)
textHiddenNeurons.pack(side=TOP, anchor=W, fill=X)

frame1.pack(side=LEFT, fill=BOTH)

#####################################################################################

def mlp_close():
    text = textSetPath.get()
    print(text)
    return 0

def mlp_ma():
    return 0


def rnn_close():
     _reccurentNetwork = reccurentNetwork()
     _reccurentNetwork.generate_sets(set_path,set_name)
     _reccurentNetwork.train_network()
     _reccurentNetwork.test_network()


def rnn_ma():
    return 0


def validation():
    return 0

#####################################################################################
button_frame=Frame(main)

_btnMLPclose = tkinter.Button(button_frame, text="MLP close",command=mlp_close)
_btnMLPclose.pack(padx=5, pady=10, side=LEFT)

_btnMLPma = tkinter.Button(button_frame, text="MLP MA",command=mlp_ma)
_btnMLPma.pack(padx=5, pady=10, side=LEFT)

_btnRNNclose = tkinter.Button(button_frame, text="RNN close",command=rnn_close)
_btnRNNclose.pack(padx=5, pady=10, side=LEFT)

_btnRNNma = tkinter.Button(button_frame, text="RNN MA",command=rnn_ma)
_btnRNNma.pack(padx=5, pady=10, side=LEFT)

_btnValidation = tkinter.Button(button_frame, text="Validation",command=validation)
_btnValidation.pack(padx=5, pady=10, side=LEFT)

button_frame.pack(side="top")

#####################################################################################

main.mainloop()
