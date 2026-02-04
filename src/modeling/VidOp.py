from .Dual import DualModel

class VidOpModel(DualModel):
    def __init__(self, config, class_names, device='cuda', **kwargs):
        super().__init__(config, class_names, device, **kwargs)
        # VidOp uses the same architecture as Dual, so no need to override methods, only it always has text context = 0
        self.config["ctx-len"] = 0