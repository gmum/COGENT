"""Build the MedGS ``Scene``, applying defaults and ``cfg_args`` overrides."""
import argparse
import os


# Sensible MedGS defaults applied when ``cfg_args`` does not set them.
MEDGS_DEFAULT_FIELDS = {
    "distance": 1.0,
    "pipeline": "img",
    "poly_degree": 1,
    "random_background": False,
    "batch_size": 1,
}


def build_medgs_args(args, extra, ModelParams, PipelineParams, sh_degree):
    """Build a Namespace compatible with MedGS ``ModelParams`` / ``PipelineParams``.

    ``extra`` is the list of unknown args returned by ``parser.parse_known_args``
    in :mod:`attack.cli`; we forward them so MedGS-specific flags still work.
    """
    parser = argparse.ArgumentParser()
    mp = ModelParams(parser)
    pp = PipelineParams(parser)
    argv = ["--model_path", args.model_path, "--sh_degree", str(sh_degree)]
    if args.source_path:
        argv += ["--source_path", args.source_path]
    argv += list(extra)
    parsed = parser.parse_args(argv)
    return parsed, mp, pp


def merge_cfg_args_and_defaults(model_params, model_path,
                                override_source_path=None):
    """Populate ``ModelParams`` from ``cfg_args`` and ``MEDGS_DEFAULT_FIELDS``.

    MedGS writes a Python ``Namespace`` literal into ``<model_path>/cfg_args``
    at training time. We restore any fields missing on ``model_params``,
    then fill in anything still missing from :data:`MEDGS_DEFAULT_FIELDS`.
    """
    cfg_path = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                content = f.read().strip()
            cfg_ns = eval(content, {"Namespace": __import__("argparse").Namespace})
            for k, v in vars(cfg_ns).items():
                if not hasattr(model_params, k):
                    setattr(model_params, k, v)
        except Exception as e:
            print(f"  WARNING: failed to parse cfg_args: {e}")

    if override_source_path:
        setattr(model_params, "source_path", override_source_path)

    for k, v in MEDGS_DEFAULT_FIELDS.items():
        if not hasattr(model_params, k):
            setattr(model_params, k, v)
    return model_params
