from torch import nn

from configs.model_config import cfg as model_cfg


class MLP(nn.Module):
    def __init__(self, cfg):
        """
        Class for building MLP model.
        :param cfg: model config
        """
        super(MLP, self).__init__()
        self.cfg = cfg

        if self.cfg.inner_layers_num == 2:
            self.fc_1 = nn.Linear(self.cfg.in_features_size, self.cfg.inner_features_size[0])
            self.fc_2 = nn.Linear(self.cfg.inner_features_size[0], self.cfg.inner_features_size[1])
            self.fc_3 = nn.Linear(self.cfg.inner_features_size[1], self.cfg.out_features_size)
            self.initialize_weight([self.fc_1, self.fc_2, self.fc_3])

        elif self.cfg.inner_layers_num == 1:
            self.fc_1 = nn.Linear(self.cfg.in_features_size, self.cfg.inner_features_size)
            self.fc_2 = nn.Linear(self.cfg.inner_features_size, self.cfg.out_features_size)
            self.initialize_weight([self.fc_1, self.fc_2])

        else:
            raise Exception

        self.drop_out_1 = nn.Dropout()
        self.drop_out_2 = nn.Dropout()
        self.drop_out_3 = nn.Dropout()
        self.relu = nn.ReLU()

    @staticmethod
    def initialize_weight(layers):
        """
        Weights Xavier initialization.
        :param layers: layers to init weights in
        """
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Forward pass.
        :param x: input vector
        :return: model output
        """
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.drop_out_1(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.drop_out_2(x)
        if self.cfg.inner_layers_num == 2:
            x = self.fc_3(x)
            x = self.relu(x)
            x = self.drop_out_3(x)
        return x


def get_model():
    """
    Gets MLP model.
    :return: MLP model
    """
    model = MLP(model_cfg)
    return model.cuda()
