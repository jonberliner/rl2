from pyt.modules import MLP, Swish

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as td

import numpy as np
from colour import hsl2rgb
import arrow

from matplotlib import pyplot as plt

from util import laggy_logistic_fn


def serve(model):
    """sample rl2 params from model and return rendered + timestamp + params"""
    t0 = arrow.now()
    hsl0, hsl1, trans01, trans10 = model()
    frames = render(hsl0, hsl1, trans01, trans10)
    return frames, t0, (hsl0, hsl1, trans01, trans10)


def render(hsl0,
           hsl1,
           trans01,
           trans10):
    """take params and return array of rendered frames to loop"""
    # TODO: decide if want to actually use hsl
    hsl0 = hsl0.unsqueeze(0)
    hsl1 = hsl1.unsqueeze(0)

    _, w_hsl0_01 = laggy_logistic_fn(trans01[0],
                                     trans01[1],
                                     trans01[2])
    w_hsl0_01 = w_hsl0_01.unsqueeze(-1)
    hsl_01 = (w_hsl0_01 * hsl0) + ((1. - w_hsl0_01) * hsl1)

    _, w_hsl0_10 = laggy_logistic_fn(trans10[0],
                                     trans10[1],
                                     trans10[2])
    w_hsl0_10 = w_hsl0_10.unsqueeze(-1)
    hsl_10 = (w_hsl0_10 * hsl1) + ((1. - w_hsl0_10) * hsl0)

    frames = torch.cat([hsl_01, hsl_10]).data.numpy()
    rgb_frames = np.stack([hsl2rgb(frame) for frame in frames])
    return rgb_frames


class RL2ParamGenerator(nn.Module):
    # model
    #   f := k -> R+
    #   k ~ Categorical
    #   colors ~ Beta(f(k), f(k))  # assuming proper col indexing of f(k)
    #   trans ~ Gamma(f(k), f(k))  # assuming proper col indexing of f(k)
    def __init__(self,
                 n_clusters,
                 cluster_to_params_graph):

        super().__init__()
        self.cluster_logits = nn.Parameter(torch.randn(n_clusters))
        self.cluster_to_params_graph = cluster_to_params_graph
        self.cluster_temperature_logit = nn.Parameter(torch.tensor(1.))
        self.cluster_distr = td.RelaxedOneHotCategorical(
                temperature=F.softplus(self.cluster_temperature_logit),
                logits=self.cluster_logits)

    def sample_params(self, n_sample=torch.Size([])):
        clusters = self.cluster_distr.rsample(n_sample)
        params = self.cluster_to_params_graph(clusters)

        alpha_hsl0 = F.softplus(params[0:3])
        beta_hsl0 = F.softplus(params[3:6])
        hsl0 = td.Beta(alpha_hsl0, beta_hsl0).rsample()

        alpha_hsl1 = F.softplus(params[6:9])
        beta_hsl1 = F.softplus(params[9:12])
        hsl1 = td.Beta(alpha_hsl1, beta_hsl1).rsample()

        shape_trans01 = F.softplus(params[12:15])
        scale_trans01 = F.softplus(params[15:18])
        trans01 = td.Gamma(shape_trans01, scale_trans01).rsample()

        shape_trans10 = F.softplus(params[18:21])
        scale_trans10 = F.softplus(params[21:24])
        trans10 = td.Gamma(shape_trans10, scale_trans10).rsample()

        return hsl0, hsl1, trans01, trans10

    def forward(self, n_sample=torch.Size([])):
        hsl0, hsl1, trans01, trans10 = self.sample_params(n_sample)
        return hsl0, hsl1, trans01, trans10




if __name__ == '__main__':
    # DETERMINE INPUT AND OUTPUT SIZES
    N_CLUSTER = 1

    n_param_per_color = 3  # hsl
    n_param_per_transition = 3  # t_before_trans, t_trans, t_after_trans

    n_param_per_color_distr = n_param_per_color * 2  # beta distr
    n_param_per_transition_distr = n_param_per_transition * 2  # gamma distr

    n_model_params = n_param_per_color * 2\
                     + n_param_per_transition * 2

    n_model_params_distr = n_param_per_color_distr * 2\
                           + n_param_per_transition_distr * 2

    # BUILD OUR RL2 MODEL
    params_net = MLP(N_CLUSTER, n_model_params_distr, [], act_fn=Swish())
    param_generator = RL2ParamGenerator(N_CLUSTER, params_net)

    # opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
    #                        lr=1e-1)


    times = [arrow.now()]

    def press(event):
        serve_another = event.key == ' '
        print(event.key)
        if serve_another:
            times.append(arrow.now())
            timer_y = (times[-2] - times[-1]).total_seconds()

            # # update watch-time regressor
            # model.add_data(torch.cat(list(params), 1), timer_y)
            # loss = torch.tensor(timer_y + 1.).pow(-1)
            # update(estimate_opt, estimate_loss)

            params = param_generator()
            frames = render(*params)


            n_frame = len(frames)
            i_frame = 0
            while True:
                i_frame = (i_frame + 1) % n_frame
                im.set_data(frames[i_frame][None, None, :])
                ax.set_title(i_frame)
                fig.canvas.draw()
                plt.pause(PAUSE)

        else:
            ax.set_title('hit spacebar to serve another')

    fig, ax = plt.subplots()
    cid = fig.canvas.mpl_connect('key_press_event', press)

    INIT_IM = np.zeros([1, 1, 3])
    im = ax.imshow(INIT_IM)

    ax.set_title('hit spacebar to serve another')
    PAUSE = 1. / 24.
    plt.show()
