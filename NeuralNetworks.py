from help_functions import *
import os

def check_correct_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

class MLP:

    def run_network(self,set_path,set_name,save_results_path,epochs,hidden_neurons,expected_error=0):
        sets = generate_sets(set_path, set_name)
        ds = generate_supervised_data_set(sets[0], sets[1])
        n = generate_mlp_network(input_neurons=len(sets[0][0]), hidden_neurons=hidden_neurons, output_neurons=1)
        trained_network, train_error = train_network(network=n, data_set=ds, epochs=epochs,
                                                     expected_train_error=expected_error)
        check_correct_path(save_results_path)
        error_train, error_val = generate_plots_and_errors(save_results_path=save_results_path, set_name=set_name,
                                                           network=trained_network, train_errors=train_error, sets=sets)
        return error_train, error_val


class RNN:
    def run_network(self,set_path,set_name,save_results_path,epochs,hidden_neurons,expected_error=0):
        sets = generate_sets(set_path, set_name)
        ds = generate_sequential_data_set(sets[0], sets[1])
        n = generate_rnn_network(input_neurons=len(sets[0][0]), hidden_neurons=hidden_neurons, output_neurons=1)
        trained_network, train_error = train_network(network=n, data_set=ds, epochs=epochs,
                                                     expected_train_error=expected_error)
        check_correct_path(save_results_path)
        error_train, error_val = generate_plots_and_errors(save_results_path=save_results_path, set_name=set_name,
                                                           network=trained_network, train_errors=train_error, sets=sets)
        return error_train, error_val