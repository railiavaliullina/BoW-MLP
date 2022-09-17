from easydict import EasyDict


cfg = EasyDict()

cfg.batch_size = 128
cfg.lr = 0.001  # 1e-4
cfg.weight_decay = 1e-4
cfg.epochs = 5  # 20  # 100

cfg.log_metrics = True
cfg.experiment_name = 'lr_0.001_SGD_1_hidden_layer_128_dim'

cfg.evaluate_on_train_set = True
cfg.evaluate_before_training = True
cfg.eval_plots_dir = f'../saved_files/plots/{cfg.experiment_name}'
cfg.plot_conf_matrices = True

cfg.load_saved_model = False
cfg.checkpoints_dir = f'../saved_files/checkpoints/{cfg.experiment_name}'
cfg.epoch_to_load = 9
cfg.save_model = False
cfg.epochs_saving_freq = 1
