# Model Weights Directory

Place your clinically trained Diabetic Retinopathy PyTorch weights here:
`weights/dr_model.pt`

### Expected Architecture:
- Backbone: `torchvision.models.efficientnet_b0`
- Classifier Head: `nn.Linear(in_features=1280, out_features=5)`
- Classes:
  - 0: No DR
  - 1: Mild
  - 2: Moderate
  - 3: Severe
  - 4: Proliferative

When `dr_model.pt` is missing, **NetraSathi** automatically runs in **DEMO MODE** using an ImageNet-initialized backbone and clearly marks all outputs, dashboards, and PDF reports as illustrative college demo predictions.
