
#
# Created in 2022 by Gaëtan Serré
#

# Clone the original ViT repository
import os
os.system("[ -d vision_transformer ] || git clone --depth=1 https://github.com/google-research/vision_transformer")

# Install the dependencies
import sys
sys.executable
os.system(sys.executable + " -m pip install -qr vision_transformer/vit_jax/requirements.txt")

# Import files from repository.

# Provide OOM errors
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import sys
if './vision_transformer' not in sys.path:
  sys.path.append('./vision_transformer')


from tqdm.auto import tqdm
import tensorflow as tf
import jax

from vit_jax import checkpoint
from vit_jax import models
from vit_jax.configs import models as models_config

# Inference functions

def get_run_model_jit(model, params):
  return jax.jit(lambda x: model.apply({"params": params}, x, train=False))

def get_accuracy(model, params, dataset):
  run_model_jit = get_run_model_jit(model, params)

  good = 0
  total = 0
  tqdm_loader = tqdm(dataset, desc="Accuracy", unit="batches")
  for images, labels in tqdm_loader:
    images = images.numpy() / 255.0
    logits = run_model_jit(images)

    good  += jax.device_get(jax.numpy.equal(logits.argmax(axis=1), labels.numpy()).sum())
    total += labels.shape[0]

    tqdm_loader.set_description_str(f"Accuracy: {good / total:.2%}")

  return good / total

# Load the ImageNet validation split

# Force tensorflow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ds = tf.keras.utils.image_dataset_from_directory(
  "ImageNet/val",
  batch_size=16,
  image_size=(384, 384),
  shuffle=False
)

# For jax to use GPU
del os.environ["CUDA_VISIBLE_DEVICES"]

# Evaluate the original ViT-B/16 & ViT-L/16 models the provided weights

def evaluate(model_name):
  model_config = models_config.MODEL_CONFIGS[model_name]
  path = f"gs://vit_models/imagenet21k+imagenet2012/{model_name}.npz"
  print(f'{tf.io.gfile.stat(path).length / 1024 / 1024:.1f} MiB - {path}')
  
  model  = models.VisionTransformer(num_classes=1000, **model_config)
  params = checkpoint.load(path)

  return get_accuracy(model, params, ds)

accuracy_b_16 = evaluate("ViT-B_16")
print(f"ViT-B/16 with Imagenet21k accuracy: {accuracy_b_16:.2%}")

accuracy_l_16 = evaluate("ViT-L_16")
print(f"ViT-L/16 with Imagenet21k accuracy: {accuracy_l_16:.2%}")
