from tkinter import *
import tkinter
from Validation import *
from NeuralNetworks import *
import os

#set_path = "/home/adminz/repo/mgr/sets/"
#save_results_path = "/home/adminz/results"
#set_name = "close"


#####################################################################################

main = tkinter.Tk()
main.geometry('{}x{}'.format(600,250))
main.resizable(width=False, height=False)

#####################################################################################

frame1=Frame(main)

label5 = tkinter.Label(frame1,text="set path")
label5.pack(side=TOP, anchor=W, fill=X)
textSetPath = tkinter.Entry(frame1)
textSetPath.insert(END, '/home/adminz/repo/mgr/sets/')
textSetPath.pack(side=TOP, anchor=W, fill=X)

label6 = tkinter.Label(frame1,text="path to save results")
label6.pack(side=TOP, anchor=W, fill=X)
textSaveResults = tkinter.Entry(frame1)
textSaveResults.insert(END, '/home/adminz/results/')
textSaveResults.pack(side=TOP, anchor=W, fill=X)

label7 = tkinter.Label(frame1,text="set name")
label7.pack(side=TOP, anchor=W, fill=X)
textSetName = tkinter.Entry(frame1)
textSetName.insert(END, 'close')
textSetName.pack(side=TOP, anchor=W, fill=X)

label1=tkinter.Label(frame1,text="Iterations")
label1.pack(side=TOP, anchor=W, fill=X)
textIterations = tkinter.Entry(frame1)
textIterations.pack(side=TOP, anchor=W, fill=X)

label2 = tkinter.Label(frame1,text="Epochs")
label2.pack(side=TOP, anchor=W, fill=X)
textEpochs = tkinter.Entry(frame1)
textEpochs.insert(END, '2')
textEpochs.pack(side=TOP, anchor=W, fill=X)

label3 = tkinter.Label(frame1,text="Alpha")
label3.pack(side=TOP, anchor=W, fill=X)
textAlpha = tkinter.Entry(frame1)
textAlpha.pack(side=TOP, anchor=W, fill=X)

label4 = tkinter.Label(frame1,text="Hidden Neurons")
label4.pack(side=TOP, anchor=W, fill=X)
textHiddenNeurons = tkinter.Entry(frame1)
textHiddenNeurons.insert(END, '9')
textHiddenNeurons.pack(side=TOP, anchor=W, fill=X)

frame1.pack(side=LEFT, fill=BOTH)

#####################################################################################


def check_correct_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mlp():
    try:
        set_path = textSetPath.get()
        set_name = textSetName.get()
        save_results_path = textSaveResults.get() + 'mlp_' + set_name

        epochs = int(textEpochs.get())
        hidden_neurons = int(textHiddenNeurons.get())
        _mlp = MLP()
        _mlp.run_network(set_path, set_name, save_results_path, epochs, hidden_neurons)
        print('MLP end')
    except Exception as e:
        print(e)


def rnn():
    try:
        set_path = textSetPath.get()
        set_name = textSetName.get()
        save_results_path = textSaveResults.get() + 'rnn_' + set_name

        epochs = int(textEpochs.get())
        hidden_neurons = int(textHiddenNeurons.get())
        _rnn = RNN()
        _rnn.run_network(set_path, set_name, save_results_path, epochs, hidden_neurons)
        print('RNN end')
    except Exception as e:
        print(e)


def validation():
    try:
        print('Start validate')
        _validate = VALIDATE()
        set_path = textSetPath.get()
        save_results_path = textSaveResults.get()
        save_results_path = save_results_path + 'validation/'
        check_correct_path(save_results_path)

        _validate.search_network(set_path=set_path, save_results_path=save_results_path)
        print('Stop validate')
    except Exception as e:
        print(e)


#####################################################################################
button_frame=Frame(main)

_btnMLPclose = tkinter.Button(button_frame, text="MLP close",command=mlp)
_btnMLPclose.pack(padx=5, pady=10, side=LEFT)

_btnMLPma = tkinter.Button(button_frame, text="MLP MA")
_btnMLPma.pack(padx=5, pady=10, side=LEFT)

_btnRNNclose = tkinter.Button(button_frame, text="RNN close",command=rnn)
_btnRNNclose.pack(padx=5, pady=10, side=LEFT)

_btnRNNma = tkinter.Button(button_frame, text="RNN MA")
_btnRNNma.pack(padx=5, pady=10, side=LEFT)

_btnValidation = tkinter.Button(button_frame, text="Validation",command=validation)
_btnValidation.pack(padx=5, pady=10, side=LEFT)

button_frame.pack(side="top")

#####################################################################################

main.mainloop()
