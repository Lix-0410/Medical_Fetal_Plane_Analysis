const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const previewImg = document.getElementById('previewImg');
const scannerContent = document.getElementById('scannerContent');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const loadingState = document.getElementById('loadingState');
const loadingText = document.getElementById('loadingText');
const resultsEl = document.getElementById('results');

const stage1Block = document.getElementById('stage1Block');
const stage1Tag = document.getElementById('stage1Tag');
const stage1Fill = document.getElementById('stage1Fill');
const stage1Num = document.getElementById('stage1Num');

const stage2Block = document.getElementById('stage2Block');
const stage2Tag = document.getElementById('stage2Tag');
const confidenceList = document.getElementById('confidenceList');

const gradcamBlock = document.getElementById('gradcamBlock');
const originalImg = document.getElementById('originalImg');
const overlayImg = document.getElementById('overlayImg');

let selectedFile = null;

const READABLE_NAMES = {
  'Brain_Trans_cerebellum': 'Brain — Trans-cerebellum',
  'Brain_Trans_thalamic': 'Brain — Trans-thalamic',
  'Brain_Trans_ventricular': 'Brain — Trans-ventricular',
  'Fetal_abdomen': 'Fetal abdomen',
  'Fetal_femur': 'Fetal femur',
  'Fetal_thorax': 'Fetal thorax',
  'Maternal_cervix': 'Maternal cervix',
  'Other': 'Other / Non-standard'
};

function readableName(key) {
  return READABLE_NAMES[key] || key;
}

// ── Dropzone interactions ──
dropzone.addEventListener('click', () => fileInput.click());
dropzone.setAttribute('tabindex', '0');
dropzone.setAttribute('role', 'button');
dropzone.setAttribute('aria-label', 'Upload an ultrasound image');
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

['dragover', 'dragenter'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'rgba(63, 169, 245, 0.6)';
  });
});

['dragleave', 'drop'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.style.borderColor = '';
  });
});

dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  if (!file.type.startsWith('image/')) return;
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.style.display = 'block';
    scannerContent.style.display = 'none';
  };
  reader.readAsDataURL(file);

  analyzeBtn.disabled = false;
  resetBtn.style.display = 'inline-block';
  resultsEl.hidden = true;
}

resetBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  selectedFile = null;
  previewImg.src = '';
  previewImg.style.display = 'none';
  scannerContent.style.display = 'flex';
  analyzeBtn.disabled = true;
  resetBtn.style.display = 'none';
  resultsEl.hidden = true;
  fileInput.value = '';
});

// ── Analyze ──
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  resultsEl.hidden = true;
  loadingState.hidden = false;
  analyzeBtn.disabled = true;
  loadingText.textContent = 'Running Stage 1 — checking for a standard plane…';

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const response = await fetch('/predict', { method: 'POST', body: formData });
    if (!response.ok) throw new Error('Prediction failed');
    const data = await response.json();

    loadingText.textContent = 'Rendering results…';
    setTimeout(() => renderResults(data), 250);
  } catch (err) {
    loadingState.hidden = true;
    analyzeBtn.disabled = false;
    alert('Something went wrong analyzing this image. Please try again.');
    console.error(err);
  }
});

function renderResults(data) {
  loadingState.hidden = true;
  resultsEl.hidden = false;

  const isOther = data.final_label === 'Other';
  const s1Confidence = data.stage1_confidence;

  // ── Stage 1 ──
  stage1Tag.textContent = isOther ? 'Non-standard' : 'Standard plane detected';
  stage1Tag.classList.toggle('is-other', isOther);

  const s1Pct = Math.round(s1Confidence * 100);
  stage1Fill.style.width = s1Pct + '%';
  stage1Fill.classList.toggle('is-other', isOther);
  stage1Num.textContent = s1Pct + '%';

  // ── Stage 2 ──
  if (!isOther && data.stage2_confidences) {
    stage2Block.hidden = false;
    stage2Tag.textContent = readableName(data.final_label);

    const sorted = Object.entries(data.stage2_confidences).sort((a, b) => b[1] - a[1]);
    confidenceList.innerHTML = '';
    sorted.forEach(([cls, prob], idx) => {
      const pct = Math.round(prob * 100);
      const row = document.createElement('div');
      row.className = 'confidence-row' + (idx === 0 ? ' is-top' : '');
      row.innerHTML = `
        <span class="cls-name">${readableName(cls)}</span>
        <div class="confidence-bar"><div class="confidence-fill" style="width:${pct}%"></div></div>
        <span class="confidence-num">${pct}%</span>
      `;
      confidenceList.appendChild(row);
    });
  } else {
    stage2Block.hidden = true;
  }

  // ── Grad-CAM ──
  if (data.original_image && data.gradcam_overlay) {
    gradcamBlock.hidden = false;
    originalImg.src = 'data:image/png;base64,' + data.original_image;
    overlayImg.src = 'data:image/png;base64,' + data.gradcam_overlay;
  } else {
    gradcamBlock.hidden = true;
  }

  resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
  analyzeBtn.disabled = false;
}
