# Kaggle Training Export

This folder contains the files needed to train the model on Kaggle's free GPU instances.

## Files
1.  **dataset.zip**: The full merged dataset (2900 images). Upload this to Kaggle.
2.  **kaggle_train_script.py**: Python code to copy-paste into your Kaggle Notebook.

## Instructions

1.  **Get the Data**:
    *   The `dataset.zip` file is being generated in this folder.
    *   Go to [Kaggle Datasets](https://www.kaggle.com/datasets) -> "New Dataset".
    *   Upload `dataset.zip`. Name it something like `doclayout-dataset`.

2.  **Create Notebook**:
    *   Go to [Kaggle Kernels](https://www.kaggle.com/code) -> "New Notebook".
    *   **Settings** (Sidebar):
        *   Accelerator: **GPU T4** (or better).
        *   Internet: **On** (for installing ultralytics).
    *   **Add Data** (Sidebar):
        *   Click "Add Data".
        *   Select "Your Datasets" -> `doclayout-dataset`.

3.  **Run and Forget (Background Execution)**:
    *   Open `kaggle_train_script.py` locally and copy the code into your Kaggle Notebook.
    *   Click **"Save Version"** (top right).
    *   Select **"Save & Run All (Commit)"**.
    *   Click **"Save"**.
    *   🎉 **Done!** You can close the tab. Kaggle will run it in the background for up to 9 hours.

4.  **Download Results**:
    *   Check back in ~1 hour.
    *   Go to the Notebook's **"Data"** or **"Output"** tab (view mode).
    *   Download `results.zip` or the `best.pt` model.
