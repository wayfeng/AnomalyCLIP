import torch
import openvino as ov
from prompt_ensemble import AnomalyCLIP_PromptLearner
import AnomalyCLIP_lib

class ImageEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model

    def forward(self, image):
        with torch.no_grad():
            image_features, _ = self.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features

class TextEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model

    def forward(self, prompts, tokenized_prompts, compound_prompts_text):
        return self.model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text)

def load_prompt_learner(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    checkpoint['prompt_learner']['compound_prompts_text'] = torch.stack([
            checkpoint['prompt_learner'][f'compound_prompts_text.{i}'] for i in range(depth-1)])
    for i in range(depth-1):
        del checkpoint['prompt_learner'][f'compound_prompts_text.{i}']
    prompt_learner = AnomalyCLIP_PromptLearner(model, parameters)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    return prompt_learner

if __name__ == '__main__':
    depth = 9
    parameters = {
        "Prompt_length": 12,
        "learnabel_text_embedding_depth": depth,
        "learnabel_text_embedding_length": 4
    }
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device='cpu', design_details = parameters)
    model.eval()
    model.visual.DAPM_replace(DPAM_layer = 20)

    sz = 518 # convert_model only works with image size 336
    ie = ImageEncoder(model)
    traced_model = torch.jit.trace(
        ie,
        example_inputs=torch.randn(1,3,sz,sz),
        strict=False,
        check_trace=False,
    )
    ov_ie = ov.convert_model(traced_model, example_input=torch.randn(1,3,sz,sz), input=(1,3,sz,sz))
    ov.save_model(ov_ie, './models/image_encoder.xml')

    checkpoint_path = './checkpoints/9_12_4_multiscale/epoch_15.pth'
    prompt_learner = load_prompt_learner(checkpoint_path)
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
    te = TextEncoder(model)
    ov_te = ov.convert_model(te, example_input={
        'prompts': prompts,
        'tokenized_prompts': tokenized_prompts,
        'compound_prompts_text': compound_prompts_text.data},
        input=([2, 77, 768], [2, 77], [8, 4, 768]))
    ov.save_model(ov_te, './models/text_encoder.xml')
