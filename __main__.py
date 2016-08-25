from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout

Window.clearcolor = (1, 1, 1, 1)
"""
class ForexNeuralNetworks(App):
    def setOrientation(self, orient):
        self.orient = orient

    def build(self):
        layout = BoxLayout(pos=(0,570))
        btn_mlp_close = Button(text="MLP Close",size_hint=(.2, 0.05))
        btn_mlp_ma = Button(text="MLP MA",size_hint=(.2, 0.05))
        btn_rnn_close = Button(text="RNN Close",size_hint=(.2, 0.05))
        btn_rnn_ma = Button(text="RNN MA",size_hint=(.2, 0.05))
        btn_contribution_chart = Button(text="Contribution Chart",size_hint=(.2, 0.05))
        btn_validation = Button(text="Validation",size_hint=(.2, 0.05))
        layout.add_widget(btn_mlp_close)
        layout.add_widget(btn_mlp_ma)
        layout.add_widget(btn_rnn_close)
        layout.add_widget(btn_rnn_ma)
        layout.add_widget(btn_contribution_chart)
        layout.add_widget(btn_validation)
        return layout
    def build(self):
        textinput = TextInput(text='Hello world')
        return textinput

_fnn = ForexNeuralNetworks()
_fnn.setOrientation(orient="vertical")
_fnn.run()
"""

class ForexNeuralNetworks(App):

    def build(self):
        return FloatLayout()
    def do_login(self, *args):
        self.txt_inpt.text="aaaa"



ForexNeuralNetworks().run()