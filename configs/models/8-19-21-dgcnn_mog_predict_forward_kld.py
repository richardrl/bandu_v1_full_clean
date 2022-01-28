# cvae_bce_vamp

from models.dgcnn_cvaeseg_model import DGCNNCVAESeg

embedding_dim = 512
latent_dim = 24
config = dict(
    model0=dict(model_class=DGCNNCVAESeg,
                model_kwargs=dict(
                    encoder_kwargs=dict(n_knn=20,
                                        num_class=embedding_dim,
                                        normal_channel=False,
                                        num_channels=4),
                    decoder_kwargs=dict(n_knn=20,
                                        num_part=1,
                                        # this is equal to 1 for binary classification per point
                                        normal_channel=False,
                                        num_channels=3 + latent_dim),
                    embedding_dim=embedding_dim,
                    latent_dim=latent_dim,
                    normalize=True,
                    prior_type="mog_predict_unit_logvar",
                    prior_predictor_type="cls",
                    prior_predictor_kwargs=dict(),
                    num_mog_prior_components=5,
                    multi_gpu=True,
                    logvar_upper_bound=2,
                    logvar_lower_bound=-6,
                    start_temperature=.5,
                    # temperature_drop_idx=0,
                    # anneal_rate=0.00003,
                    # temp_min=0.5,
                    # current_batch_idx=600*100*2,
                    sample_prior_bool=True,
                    use_fixed_temp=True
                ),
                model_name="surface_classifier"),
)
