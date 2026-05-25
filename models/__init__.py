"""Model registry and builder for SA-VSSM."""

from .vmamba import VSSM

__all__ = ["VSSM", "build_vssm_model", "build_model"]


def build_vssm_model(config, is_decoder=False):
    """Build a VSSM model from config."""
    model = VSSM(
        patch_size=config.MODEL.VSSM.PATCH_SIZE,
        in_chans=config.MODEL.VSSM.IN_CHANS,
        depths=config.MODEL.VSSM.DEPTHS,
        dims=config.MODEL.VSSM.EMBED_DIM,
        ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
        ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
        ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
        ssm_dt_rank=(
            "auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto"
            else int(config.MODEL.VSSM.SSM_DT_RANK)
        ),
        ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
        ssm_conv=config.MODEL.VSSM.SSM_CONV,
        ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
        ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
        ssm_init=config.MODEL.VSSM.SSM_INIT,
        forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
        mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
        mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
        mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
        transformerDecoder=is_decoder,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        patch_norm=config.MODEL.VSSM.PATCH_NORM,
        norm_layer=config.MODEL.VSSM.NORM_LAYER,
        downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
        patchembed_version=config.MODEL.VSSM.PATCHEMBED,
        gmlp=config.MODEL.VSSM.GMLP,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    )
    return model


def build_model(config, is_decoder=False):
    """Build a model from config. Currently only VSSM is supported."""
    if config.MODEL.TYPE in ("vssm", "vim"):
        return build_vssm_model(config, is_decoder=is_decoder)
    raise ValueError(f"Unsupported model type: {config.MODEL.TYPE}")
