from supervised_training.utils.training_util import get_batch_from_state


def neural_policy(input_state, sort_categories_by_size=False, stats_dic=None):
    batch = get_batch_from_state(s, model_device, num_points=2048, stats_dic=stats_dic)