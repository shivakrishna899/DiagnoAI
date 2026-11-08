# DiagnoAI

**A multimodal skin-lesion research prototype combining image features
and reported symptoms.**

DiagnoAI is an experimental machine-learning project that explores
whether image features and a small set of symptom inputs can be combined
to rank six skin-lesion categories. It includes a Streamlit interface, a
FastAPI inference service, model-training utilities, uncertainty
indicators, and visual explanation tools.

> \[!WARNING\] **Research use only. Not a medical device.** DiagnoAI is
> not clinically validated and must not be used to diagnose, treat, or
> rule out skin cancer or any other condition. Model scores are not
> clinical probabilities. A low score does not rule out disease, and a
> high score does not confirm it. Seek advice from a qualified
> healthcare professional about concerning or changing skin lesions.

## Project status

-   The application and multimodal inference pipeline are implemented.
-   The current selected model is the E2 ResNet-18 late-fusion
    checkpoint.
-   The E2 model was selected using validation macro-F1.
-   The final held-out E2 test evaluation has not yet been completed. Do
    not describe validation results as test performance.
-   Trained model weights and the PAD-UFES-20 image dataset are not
    included in this repository.

## Features

-   **Image branch:** ResNet-18 image encoder using 224 × 224 RGB input
    and ImageNet normalization.
-   **Symptom branch:** a small MLP that encodes six symptom fields.
-   **Late fusion:** combines the 512-dimensional image representation
    with the symptom representation before classification.
-   **Ranked outputs:** returns the model's leading class candidates.
-   **Temperature scaling:** applies a scalar fitted on validation
    logits.
-   **Uncertainty indicators:** surfaces confidence, top-candidate
    margin, entropy, and related warning signals.
-   **Grad-CAM:** highlights image regions that influenced a selected
    model output.
-   **Symptom sensitivity:** compares model outputs after changing one
    known symptom input. This is model sensitivity, not causal or
    clinical evidence.
-   **Interfaces:** Streamlit user interface and FastAPI endpoints.

## Supported labels

The current PAD-UFES-20 experiment uses these six dataset labels:

  Label   Dataset category
  ------- -------------------------
  `ACK`   Actinic keratosis
  `BCC`   Basal cell carcinoma
  `MEL`   Melanoma
  `NEV`   Melanocytic nevus
  `SCC`   Squamous cell carcinoma
  `SEK`   Seborrheic keratosis

These are dataset classes, not a complete list of skin conditions. The
model cannot reliably identify conditions outside its training
distribution.

## Model inputs

The model expects an image and six symptom values collected for the same
lesion:

-   Itching
-   Growth
-   Pain
-   Appearance change
-   Bleeding
-   Elevation

Each symptom is represented with two features so that **yes**, **no**,
and **unknown** remain distinct:

  Response   Two-feature encoding
  ---------- ----------------------
  Yes        `[1, 1]`
  No         `[0, 1]`
  Unknown    `[0, 0]`

The resulting symptom vector contains 12 values. Keep the feature names
and order defined in `diagnoai/constants.py` unchanged when preparing
data or calling inference code. Do not replace unknown values with no.

## Repository layout

``` text
.
├── api/                 # FastAPI application
├── data/                # Dataset preparation assets and templates
├── diagnoai/             # Model, inference, constants and explainability code
├── models/               # Local model artifacts (weights are git-ignored)
├── tests/                # Automated tests
├── training/             # Dataset preparation, training and evaluation code
├── app.py                # Streamlit application
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

The exact files in each folder may evolve. Dataset files, processed
images, and model checkpoints should remain outside version control.

## Requirements

-   Python 3.10 or newer is recommended.
-   Windows PowerShell instructions are shown below.
-   Install dependencies from `requirements.txt`.
-   A CPU can run the application, but model inference may be slower. A
    compatible GPU is useful for training.

## Setup

From the repository root:

``` powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use the appropriate
local execution-policy guidance for your machine, or run the
environment's Python executable directly.

## Add the model checkpoint

Model weights are intentionally excluded from Git because they are large
binary artifacts and may have separate provenance or distribution
conditions.

The Streamlit application is configured to load:

``` text
models/diagnoai_e2.pt
```

Place your authorized E2 checkpoint at that path before starting
inference. Do not rename an E1 or unrelated checkpoint to
`diagnoai_e2.pt`. The checkpoint must match the model architecture,
class order, symptom feature order, and preprocessing expected by the
inference code.

The repository does not ship the checkpoint. A fresh clone therefore
cannot run model predictions until the correct checkpoint is supplied.

## Run the application

Start the Streamlit interface:

``` powershell
streamlit run app.py
```

Start the API in a separate terminal:

``` powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

When the API is running, its interactive documentation is normally
available at:

-   `http://127.0.0.1:8000/docs`

Use the health route implemented by `api/main.py` to check service
status. Endpoint names and request schemas should be taken from the
current API source, not assumed from this README.

## Dataset and data preparation

The current research experiment uses **PAD-UFES-20**, a public
skin-lesion dataset containing 2,298 smartphone images from 1,373
patients and six diagnostic categories. The project uses image records
alongside available clinical fields for the six symptom inputs.

Dataset source:

-   Pessanha et al., *PAD-UFES-20: A skin lesion dataset composed of
    patient data and clinical images collected from smartphones*. Data
    in Brief (2020). <https://doi.org/10.1016/j.dib.2020.106221>
-   [PAD-UFES-20 dataset
    record](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

**Dataset terms are separate from this repository's software license.**
Obtain the dataset from its canonical source, review and follow the
applicable terms, cite the original dataset paper, and do not commit or
redistribute the dataset images or metadata unless the applicable terms
expressly permit it. The MIT license in this repository does not grant
rights to the dataset.

Before training, confirm that each image and symptom row genuinely
refers to the same patient and lesion. Do not invent symptom values or
treat missing symptom values as negative responses. Avoid including
direct identifiers in derived files.

The training pipeline expects a CSV with image paths, diagnosis labels,
patient identifiers, and the symptom columns required by the training
code. Use the included preparation utilities and inspect their
command-line help before running them:

``` powershell
python -m training.prepare_padufes --help
python -m training.train --help
```

Example structure for a prepared row:

``` csv
image_path,patient_id,diagnosis
path/to/image.jpg,patient-group-id,BCC
```

This abbreviated example only illustrates the path, patient grouping,
and label fields. It is **not** a complete training CSV: the required
symptom columns must also be present, as defined by the current training
implementation.

## Data splitting and evaluation

Patient-level separation is important because images from the same
patient can be highly related. The experiment uses a patient-disjoint
split:

  Partition            Images    Patients
  --------------- ----------- -----------
  Training              1,468         878
  Validation              371         220
  Held-out test           459         275
  **Total**         **2,298**   **1,373**

The split manifest is used to preserve the same assignments across
experiments. Keep it with the experiment records and do not regenerate
it when reproducing the selected model.

E2 was selected using validation macro-F1. Its validation results are
recorded in the model metrics artifact. The held-out E2 test set has not
yet been evaluated, so this README intentionally makes no E2
test-performance claim. Complete the planned final evaluation once,
without tuning against the held-out test results, before publishing test
metrics.

When reporting results, include more than accuracy. Report macro-F1 and
per-class precision/recall, along with suitable calibration metrics such
as negative log-likelihood, Brier score, and expected calibration error.
Explain the split protocol and limitations. Validation calibration does
not establish clinical calibration or external validity.

## Interpretation and limitations

-   A displayed score is a model output for the supplied inputs, not the
    probability that a person has a disease.
-   Grad-CAM is a visualization of model attention-related gradients,
    not proof that the highlighted region is medically meaningful.
-   Symptom sensitivity describes how the model output changes under an
    input perturbation. It does not establish that a symptom caused the
    prediction.
-   Temperature scaling can adjust the sharpness of model scores. It
    cannot fix biased data, distribution shift, incorrect labels, or a
    flawed model.
-   The model has not been shown to generalize to other hospitals,
    devices, populations, image types, or skin tones.
-   The dataset is limited in size and scope. Class imbalance, selection
    bias, label noise, and subgroup performance require further study.
-   Do not use this prototype for triage, treatment decisions,
    screening, or reassurance.

## Tests

Run the test suite from the repository root:

``` powershell
python -m pytest -q
```

A passing test suite checks software behavior covered by those tests. It
does not validate diagnostic accuracy, clinical safety, or real-world
performance.

## Privacy and security

Use only data you are authorized to process. Do not commit patient
images, raw datasets, patient-level manifests containing identifiable
information, credentials, API keys, or private configuration files.
Review all files before publishing the repository. A local research demo
is not a production-ready health service.

## Related research

The project review considered related open-source work on multimodal
disease prediction, skin-lesion classification, PAD-UFES-20, Grad-CAM,
and uncertainty estimation. Those repositories are references only. This
project does not include their source code or model weights. Review each
upstream repository's current license and terms before reusing any
material.

## License

The DiagnoAI source code in this repository is released under the MIT
License. See [LICENSE](LICENSE). This license does not cover third-party
datasets, pretrained weights, or other assets that may have separate
terms.
