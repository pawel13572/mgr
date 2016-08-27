from NeuralNetworks import *
import pandas as pd
from datetime import datetime


class VALIDATE:
    def search_network(self,save_results_path,set_path):
        closeMlpTrainTab = []
        closeMlpValTab = []
        closeMlpError = []

        empty = []

        maMlpTrainTab = []
        maMlpValTab = []
        maMlpError = []

        closeRecTrainTab = []
        closeRecValTab = []
        closeRecError = []

        maRecTrainTab = []
        maRecValTab = []
        maRecError = []

        Layers = []

        a = datetime.now()

        _mlp = MLP()
        _rnn = RNN()
        expected_error = 0.0005
        for i in range(2, 101): #100
            Layers.append(i)

            error_train, error_val = _mlp.run_network(set_path, 'close', save_results_path + 'mlp_close' + str(i),
                                                      epochs=10,hidden_neurons=i,expected_error=expected_error)

            closeMlpTrainTab.append(error_train)
            closeMlpValTab.append(error_val)
            closeMlpError.append(math.pow(error_train - error_val, 2) / 2)

            error_train, error_val = _mlp.run_network(set_path, 'ma', save_results_path + 'mlp_ma' + str(i),
                                                      epochs=10,hidden_neurons=i,expected_error=expected_error)

            maMlpTrainTab.append(error_train)
            maMlpValTab.append(error_val)
            maMlpError.append(math.pow(error_train - error_val, 2) / 2)

            error_train, error_val = _rnn.run_network(set_path, 'close', save_results_path + 'rnn_close' + str(i),
                                                      epochs=10,hidden_neurons=i,expected_error=expected_error)

            closeRecTrainTab.append(error_train)
            closeRecValTab.append(error_val)
            closeRecError.append(math.pow(error_train - error_val, 2) / 2)

            error_train, error_val = _rnn.run_network(set_path, 'ma', save_results_path + 'rnn_ma' + str(i),
                                                      epochs=10,hidden_neurons=i,expected_error=expected_error)

            maRecTrainTab.append(error_train)
            maRecValTab.append(error_val)
            maRecError.append(math.pow(error_train - error_val, 2) / 2)

            b = datetime.now()

            c = b - a

            print("Layers: ", i, "  Time: ", c)

        table = pd.DataFrame(Layers)

        table["close_mlp_Train"] = pd.DataFrame(closeMlpTrainTab)
        table["close_mlp_Val"] = pd.DataFrame(closeMlpValTab)
        table["close_mlp_Error"] = pd.DataFrame(closeMlpError)

        table["ma_mlp_Train"] = pd.DataFrame(maMlpTrainTab)
        table["ma_mlp_Val"] = pd.DataFrame(maMlpValTab)
        table["ma_mlp_Error"] = pd.DataFrame(maMlpError)

        table["close_rec_Train"] = pd.DataFrame(closeRecTrainTab)
        table["close_rec_Val"] = pd.DataFrame(closeRecValTab)
        table["close_rec_Error"] = pd.DataFrame(closeRecError)

        table["ma_rec_Train"] = pd.DataFrame(maRecTrainTab)
        table["ma_rec_Val"] = pd.DataFrame(maRecValTab)
        table["ma_rec_Error"] = pd.DataFrame(maRecError)

        writer = pd.ExcelWriter(save_results_path+'errors.xlsx')
        table.to_excel(writer)
        writer.save()