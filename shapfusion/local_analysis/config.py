"""Parameters of the local analyses: the ``local_analysis`` block of configs/paper.yaml,
plus views / folds / paths from the rest of the configuration."""
import copy


def build(cfg: dict) -> dict:
    la = copy.deepcopy(cfg["local_analysis"])
    la["views"] = list(cfg["data"]["views"])
    la["folds"] = list(cfg["folds"])
    la["root"] = cfg["root"]
    la["out_root_path"] = cfg["out_path"] / "local_analysis"
    la["tag"] = ""
    la["paper"] = cfg
    for m, spec in la["models"].items():
        spec.update(draws=int(cfg["models"][m].get("draws", 1)),
                    logit64=bool(cfg["models"][m].get("logit64", False)))
    return la


def section(la, name):
    """Output dir of a section, <out>/local_analysis/<name>/."""
    return la["out_root_path"] / f"{name}{la.get('tag', '')}"
