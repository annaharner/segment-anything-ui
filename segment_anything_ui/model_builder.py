try:
    from efficientvit.sam_model_zoo import EfficientViTSam
    from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor, EfficientViTSamAutomaticMaskGenerator
    IS_EFFICIENT_VIT_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    import logging
    logging.warning("Efficient is not available, please install the package from https://github.com/mit-han-lab/efficientvit/tree/master .")
    IS_EFFICIENT_VIT_AVAILABLE = False

from segment_anything import sam_model_registry
from segment_anything import SamPredictor, automatic_mask_generator
from segment_anything.build_sam import Sam


def build_model(model_name: str, checkpoint_path: str, device: str):
    match model_name:
        # case "xl0" | "xl1":
        #     efficientvit_sam = create_sam_model(
        #         name=model_name, weight_url=checkpoint_path,
        #     )
        #     return efficientvit_sam.to(device).eval()
        
        case "vit_b" | "vit_l" | "vit_h":
            sam = sam_model_registry[model_name](
                checkpoint=checkpoint_path)
            sam.eval()
            return sam.to(device)

        case _:
            raise ValueError(f"Model {model_name} not supported")
        
        
def get_predictor(sam: EfficientViTSam):
    if isinstance(sam, EfficientViTSam):
        return EfficientViTSamPredictor(sam)
    elif isinstance(sam, Sam):
        return SamPredictor(sam)
    else:
        raise ValueError("Model is not an EfficientViTSam or Sam")
    
def get_mask_generator(sam: EfficientViTSam, **kwargs):
    if isinstance(sam, EfficientViTSam):
        return EfficientViTSamAutomaticMaskGenerator(model=sam, **kwargs)
    elif isinstance(sam, Sam):
        return automatic_mask_generator.SamAutomaticMaskGenerator(model=sam, **kwargs)
    else:
        raise ValueError("Model is not an EfficientViTSam or Sam")