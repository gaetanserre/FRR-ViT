# Clone the original ViT repository
import os
os.system("[ -d vision_transformer ] || git clone --depth=1 https://github.com/google-research/vision_transformer")

# Install the dependencies
import sys
sys.executable
os.system(sys.executable + " -m pip install -qr vision_transformer/vit_jax/requirements.txt")

import sys
if './vision_transformer' not in sys.path:
  sys.path.append('./vision_transformer')

import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import jax
import optax
from random import shuffle

from vit_jax import models
from vit_jax.configs import models as models_config

from custom_vit import vit_b_16

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

NUM_CLASSES = 1000
NUM_EPOCHS = 150
NUM_ITER = 1

def test_model_torch(model, nb_epochs=3):
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.1)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model.to(device)

  model.train()
  loss_per_epoch = []
  for epoch in range(nb_epochs):
    losses = []
    for img, label in train_loader:
      img, label = img.to(device), label.to(device)

      optimizer.zero_grad()
      output = model(img)
      loss = criterion(output, label)
      loss.backward()
      optimizer.step()

      losses.append(loss.item())

    loss_per_epoch.append(np.mean(losses))
    print(f"Epoch {epoch+1}/{nb_epochs} - Loss: {loss_per_epoch[-1]:.4f}")

  model.eval()
  accuracies = []
  for img, label in test_loader:
    img, label = img.to(device), label.to(device)

    output = model(img)
    accuracy = (output.argmax(dim=1) == label).float().mean()
    accuracies.append(accuracy.item())

  return np.mean(accuracies), loss_per_epoch

def test_model_jax(model, params, optimizer_state, nb_epochs=3):
  run_model_jit_train = jax.jit(lambda params, x: model.apply({"params": params}, x, train=True))
  run_model_jit_eval  = jax.jit(lambda params, x: model.apply({"params": params}, x, train=False))


  def train_step(params, batch, optimizer_state):
    def loss_fn(params):
      logits = run_model_jit_train(params, batch["image"])
      batch["label"] = jax.nn.one_hot(batch["label"], NUM_CLASSES)
      return criterion(logits, batch["label"]).mean()

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(params)
    updates, new_optimizer_state = optimizer.update(grad, optimizer_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_optimizer_state, loss

  def test_step(params, batch):
    logits = run_model_jit_eval(params, batch["image"])
    accuracy = (logits.argmax(axis=-1) == batch["label"]).mean()
    batch["label"] = jax.nn.one_hot(batch["label"], NUM_CLASSES)
    loss = criterion(logits, batch["label"])
    return loss, accuracy

  loss_per_epoch = []
  for epoch in range(nb_epochs):
    losses = []
    for img, label in train_loader:

      batch = {"image": img, "label": label.numpy()}

      batch["image"] = batch["image"].permute(0, 2, 3, 1).numpy()

      params, optimizer_state, loss = train_step(params, batch, optimizer_state)

      losses.append(loss)
    loss_per_epoch.append(np.mean(losses))
    print(f"Epoch {epoch+1}/{nb_epochs} - Loss: {loss_per_epoch[-1]:.4f}")
  
  accuracies = []
  for img, label in test_loader:

    batch = {"image": img, "label": label.numpy()}

    batch["image"] = batch["image"].permute(0, 2, 3, 1).numpy()

    loss, accuracy = test_step(params, batch)

    accuracies.append(accuracy)

  return np.mean(accuracies), loss_per_epoch


train_ds = torchvision.datasets.ImageFolder("ImageNet/val",
  transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Resize((224, 224))
                              ])
)

idx = list(range(10_000))
shuffle(idx)
train_ds = torch.utils.data.Subset(train_ds, idx)

len_train = int(len(train_ds) * 0.8)
train_ds, test_ds = torch.utils.data.random_split(train_ds, lengths=[len_train, len(train_ds) - len_train])


batch_size_train = 64
batch_size_test = 64

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_ds ,batch_size=batch_size_test, shuffle=True)





if __name__ == "__main__":
  print("----Our implementation----")
  custom_vit  = vit_b_16(num_classes=NUM_CLASSES)
  accuracy, our_losses = test_model_torch(custom_vit, nb_epochs=NUM_EPOCHS)
  print(f"Accuracy: {accuracy:.4f}")

  print("----Original implementation----")

  model_config = models_config.MODEL_CONFIGS["ViT-B_16"]
  vit = models.VisionTransformer(num_classes=NUM_CLASSES, **model_config)
  variables = vit.init(jax.random.PRNGKey(0), jax.numpy.ones((1, 224, 224, 3)), train=False)
  state, params = variables.pop("params")
  del variables

  criterion = optax.softmax_cross_entropy
  optimizer = optax.adamw(8e-4, b1=0.9, b2=0.999, eps=1e-6, weight_decay=0.1)
  optimizer_state = optimizer.init(params)

  accuracy, original_losses = test_model_jax(vit, params, optimizer_state, nb_epochs=NUM_EPOCHS)
  print(f"Accuracy: {accuracy:.4f}")

  # Plot losses
  dist_loss = np.abs(our_losses - original_losses)
  plt.style.use("seaborn-v0_8")
  plt.plot(range(1, len(our_losses)+1), our_losses, label=f"Our loss")
  plt.plot(range(1, len(original_losses)+1), original_losses, label=f"Original loss")
  plt.plot(range(1, len(dist_loss)+1), dist_loss, label=f"$|our\;loss - original\;loss|$")
  plt.xlabel("Epoch")
  plt.legend()
  plt.savefig("diff_losses.pdf")
