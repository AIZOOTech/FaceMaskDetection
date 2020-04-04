import sys
import torch
sys.path.append('models/')

def load_pytorch_model(model_path):
    model = torch.load(model_path)
    return model

def pytorch_inference(model, img_arr):
    y_bboxes, y_scores, = model.forward(torch.tensor(img_arr).float())
    return y_bboxes.detach().numpy(), y_scores.detach().numpy()
