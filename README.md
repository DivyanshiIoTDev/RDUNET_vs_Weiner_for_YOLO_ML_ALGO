# RDUNET_vs_Weiner_for_YOLO_ML_ALGO

This repository contains a DIV2K denoising notebook that compares a classical Wiener filter with an RDUNet (Residual Dense U-Net) denoiser.

## How to run the notebook from GitHub

GitHub can preview `.ipynb` notebooks, but the normal GitHub file viewer does **not** execute notebook cells or provide a GPU runtime. Kaggle is the recommended place to run this notebook because the dataset paths already match Kaggle's `/kaggle/input` mount.

### Kaggle import and run steps

1. **Download the notebook from GitHub:** open `rdunet-vs-weiner-on-div2k (1).ipynb`, click **Raw** or **Download raw file**, and save the `.ipynb` file locally.
2. **Create a Kaggle notebook:** go to Kaggle, select **Code** → **New Notebook**.
3. **Import the downloaded notebook:** in the Kaggle editor, use **File** → **Upload Notebook**, choose `rdunet-vs-weiner-on-div2k (1).ipynb`, and confirm the upload. If Kaggle opens a blank notebook first, the upload/import option is still under the notebook **File** menu.
4. **Attach the DIV2K dataset:** open the right-side **Add Input** / **Input** panel, search for the DIV2K dataset, and add the dataset whose folders appear as:
   - `/kaggle/input/div2k-dataset/DIV2K_train_HR`
   - `/kaggle/input/div2k-dataset/DIV2K_valid_HR`
5. **Enable GPU:** open notebook **Settings** on the right side and set **Accelerator** to a GPU option such as T4/P100. If GPU options are disabled, verify your Kaggle account/phone number and check your GPU quota.
6. **Run in order:** click **Run All** or run cells from top to bottom. The first run preprocesses and caches DIV2K images into `.npy` files; later runs skip that work unless `reset_cached_tensors = True`.
7. **Check outputs:** after training, run the metric and visualization cells. The last cell displays the requested clean → noisy → filtered sample for both Wiener and RDUNet.

### Other execution options

- **Google Colab:** open the notebook through Colab, upload or mount DIV2K, then update the dataset path cells if your DIV2K folder is not located at `/kaggle/input/div2k-dataset/` or `../input/div2k-dataset/`.
- **GitHub Codespaces/Jupyter:** start a Codespace with Python/Jupyter and install the ML dependencies, but only use this for CPU checks unless your Codespace has GPU access.

## Expected training time

The notebook is configured for `batch_size = 150`, 256×256 DIV2K patches, `rdunet_base_filters = 16`, `epochs = 25`, early stopping, and a hard `max_train_hours = 29.5` guard. Actual runtime depends on GPU type, CPU image-loading speed, and whether cached `.npy` tensors already exist.

On a Kaggle NVIDIA Tesla T4, the run is designed to complete within the 30-hour budget; for DIV2K's standard 800 training images, the training loader has about 6 batches per epoch at batch size 150. A typical first run should often finish in a few hours, while cached reruns are faster because image preprocessing is skipped. The notebook includes a time-estimate cell that reports observed elapsed time per completed epoch and projects the full training time from the current run.

## Sample display format

The final visualization cell shows the same test image in the requested workflow format:

1. **Wiener filter:** clean sample image → noisy image → clean image after filtering.
2. **RDUNet:** clean sample image → noisy image → clean image after filtering.

## Troubleshooting: `IndexError` when displaying a sample

`train_generator[index]` uses zero-based Python indexing. For example, if Kaggle prints that the generator has 42 samples, the valid indexes are `0` through `41`; `train_generator[42]` is outside the array and raises `IndexError`. The notebook sample cells now call a safe helper that falls back to the last available sample and prints a message when the requested index is too high.

If the generator has zero samples, the DIV2K dataset is probably not attached or the cached `.npy` tensors were created while the dataset path was wrong. In that case, attach the DIV2K input in Kaggle, set `reset_cached_tensors = True`, and rerun the preprocessing cells.


## Training reliability changes

The notebook now avoids three common long-run failure points: unused TensorFlow dataset blocks were removed to save RAM, the Keras generator no longer defines a `__call__` method that referenced a missing `on_epoch_end()`, and validation/test splitting uses one `split_idx` value reused by both preview generators and PyTorch loaders. RDUNet also writes periodic checkpoints every 5 epochs to `rdunet_checkpoints/rdunet_latest.pt` and `rdunet_checkpoints/rdunet_epoch_XXX.pt`, with `resume_training = True` so a Kaggle reconnect can continue from the latest backup instead of starting from epoch 1 again.
