const API_BASE = 'http://127.0.0.1:8000'
const el = id => document.getElementById(id)

// ─── History: pointer-based undo/redo ────────────────────────────────────────
// Each entry: { id, image }
let historyA = [], pointerA = -1
let historyB = [], pointerB = -1

// ─── Busy flags ───────────────────────────────────────────────────────────────
// Three groups of buttons that lock independently:
//   busyA    — all Image A controls (generate, back, forward)
//   busyB    — all Image B controls
//   busyOps  — Arithmetic + Interpolation + Clear (anything that writes to Result)
//
// When busyOps is true, Image A and B controls are also disabled so the user
// cannot change the source images while a result is being computed.
let busyA   = false
let busyB   = false
let busyOps = false

// ─── Network helper ───────────────────────────────────────────────────────────
async function postJSON(path, body) {
  const res = await fetch(API_BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

// ─── Master button-state refresh ─────────────────────────────────────────────
function refreshAllButtons() {
  const hasA  = pointerA >= 0
  const hasB  = pointerB >= 0
  const canOp = hasA && hasB

  // Any source-image operation also locks while busyOps
  const lockA = busyA || busyOps
  const lockB = busyB || busyOps

  // ── Image A ──
  el('genA').disabled  = lockA
  el('backA').disabled = lockA || pointerA < 1
  el('fwdA').disabled  = lockA || pointerA >= historyA.length - 1

  // ── Image B ──
  el('genB').disabled  = lockB
  el('backB').disabled = lockB || pointerB < 1
  el('fwdB').disabled  = lockB || pointerB >= historyB.length - 1

  // ── Arithmetic ──
  el('opAdd').disabled   = busyOps || !canOp
  el('opSubAB').disabled = busyOps || !canOp
  el('opSubBA').disabled = busyOps || !canOp

  // ── Interpolation / clear ──
  el('doInterp').disabled = busyOps || !canOp
  el('clearOut').disabled = busyOps
}

// ─── History helpers ──────────────────────────────────────────────────────────
function pushHistory(target, entry) {
  if (target === 'A') {
    historyA.push(entry)
    pointerA = historyA.length - 1
  } else {
    historyB.push(entry)
    pointerB = historyB.length - 1
  }
}

function currentA() { return pointerA >= 0 ? historyA[pointerA] : null }
function currentB() { return pointerB >= 0 ? historyB[pointerB] : null }

function navigate(target, dir) {
  if (target === 'A') {
    const next = pointerA + dir
    if (next < 0 || next >= historyA.length) return
    pointerA = next
    el('imgA').src = historyA[pointerA].image
  } else {
    const next = pointerB + dir
    if (next < 0 || next >= historyB.length) return
    pointerB = next
    el('imgB').src = historyB[pointerB].image
  }
  clearFilmstrip()
  refreshAllButtons()
}

// ─── Messaging ────────────────────────────────────────────────────────────────
function setMsg(text)       { el('msg').textContent = text }
function setInterpMsg(text) { el('interpMsg').textContent = text }
function setFormula(text)   { el('formula').textContent = text }

// ─── Generate ─────────────────────────────────────────────────────────────────
async function generate(target) {
  setMsg('')
  clearFilmstrip()
  if (target === 'A') {
    busyA = true
    el('genA').textContent = 'Generating…'
  } else {
    busyB = true
    el('genB').textContent = 'Generating…'
  }
  refreshAllButtons()

  try {
    const data = await postJSON('/generate', {})
    pushHistory(target, { id: data.latent_id, image: data.image })
    if (target === 'A') el('imgA').src = data.image
    else                el('imgB').src = data.image
    return data
  } catch (e) {
    setMsg('Generation error: ' + e.message)
    throw e
  } finally {
    if (target === 'A') { busyA = false; el('genA').textContent = 'Generate New' }
    else                { busyB = false; el('genB').textContent = 'Generate New' }
    refreshAllButtons()
  }
}

// ─── Filmstrip helpers ────────────────────────────────────────────────────────
function clearFilmstrip() {
  const c = el('interpResults')
  if (c) c.innerHTML = ''
}

function setWeightVal(v) { el('weightVal').textContent = Number(v).toFixed(2) }

// ─── Arithmetic ───────────────────────────────────────────────────────────────
async function doOp(op) {
  setMsg('')
  const a = currentA(), b = currentB()
  if (!a || !b) { setMsg('Generate both A and B first'); return }
  busyOps = true
  refreshAllButtons()
  try {
    const data = await postJSON('/arithmetic', { id_a: a.id, id_b: b.id, operation: op })
    el('imgOut').src = data.image
    const labels = {
      add:          'z_out = z_A + z_B',
      subtract_ab:  'z_out = z_A − z_B',
      subtract_ba:  'z_out = z_B − z_A'
    }
    setFormula(labels[op] || '')
  } catch (e) {
    setMsg('Operation error: ' + e.message)
  } finally {
    busyOps = false
    refreshAllButtons()
  }
}

// ─── Interpolation ────────────────────────────────────────────────────────────
async function doInterp() {
  setInterpMsg('')
  const a = currentA(), b = currentB()
  if (!a || !b) { setInterpMsg('Generate both A and B first'); return }
  busyOps = true
  refreshAllButtons()
  try {
    await getFilmstrip()
  } catch (e) {
    setInterpMsg('Interpolation error: ' + e.message)
  } finally {
    busyOps = false
    refreshAllButtons()
  }
}

async function getFilmstrip() {
  const a = currentA(), b = currentB()
  if (!a || !b) return
  const res = await postJSON('/interpolate', { id_a: a.id, id_b: b.id, steps: 7 })
  if (res && res.images) renderFilmstrip(res.images, res.ts, res.latent_ids)
}

// ─── Filmstrip rendering ──────────────────────────────────────────────────────
// Backend convention:
//   t = 0  ->  pure B (left)
//   t = 1  ->  pure A (right)
//
// Slider weight convention (what the user sees):
//   w = 1  ->  pure B   (left side of slider)
//   w = 0  ->  pure A   (right side of slider)
//   w = 1 - t
//
// Display order: we want w=1.0 (pure B) on the LEFT and w=0.0 (pure A) on
// the RIGHT, so we REVERSE the images array before rendering.
function renderFilmstrip(images, ts, latentIds) {
  const container = el('interpResults')
  if (!container) return
  container.innerHTML = ''

  // Reverse so pure-B is on the left (w=1.0) and pure-A is on the right (w=0.0)
  const reversed = images.map((src, i) => ({
    src,
    t: ts ? ts[i] : i / (images.length - 1),
    latentId: latentIds ? latentIds[i] : null
  })).reverse()

  reversed.forEach(({ src, t, latentId }) => {
    const exactWeight = 1 - t
    const displayWeight = parseFloat(exactWeight.toFixed(2))

    const fig = document.createElement('figure')
    fig.className = 'interp-thumb'

    const img = document.createElement('img')
    img.src = src
    img.alt = `w=${displayWeight.toFixed(2)}`
    img.dataset.weight = String(displayWeight)
    img.dataset.weightExact = String(exactWeight)
    if (latentId) img.dataset.latentId = String(latentId)

    img.addEventListener('click', () => {
      if (interpTimeout) {
        clearTimeout(interpTimeout)
        interpTimeout = null
      }

      // Keep slider UI in sync while showing the already-cached thumbnail image.
      el('weight').value = String(displayWeight)
      setWeightVal(displayWeight)
      el('imgOut').src = src
      setFormula(`z_out = ${exactWeight.toFixed(2)} * z_B + ${(1 - exactWeight).toFixed(2)} * z_A`)
      highlightThumb(exactWeight)
    })

    const cap = document.createElement('figcaption')
    cap.textContent = displayWeight.toFixed(2)

    fig.appendChild(img)
    fig.appendChild(cap)
    container.appendChild(fig)
  })

  highlightThumb(Number(el('weight').value))
}

function highlightThumb(weight) {
  const container = el('interpResults')
  if (!container) return
  container.querySelectorAll('.interp-thumb').forEach(f => {
    const img = f.querySelector('img')
    const w = Number(img.dataset.weightExact ?? img.dataset.weight)
    f.classList.toggle('selected', Math.abs(w - weight) < 0.005)
  })
}

// ─── Slider / weighted update ─────────────────────────────────────────────────
// Debounced: waits 120 ms after last slider movement before firing.
let interpTimeout = null
function triggerWeightedUpdate() {
  if (interpTimeout) clearTimeout(interpTimeout)
  interpTimeout = setTimeout(async () => {
    interpTimeout = null
    const w = Number(el('weight').value)
    const a = currentA(), b = currentB()
    if (!a || !b) return
    try {
      const res = await postJSON('/interpolate', { id_a: a.id, id_b: b.id, weight: w })
      if (res && res.image) {
        el('imgOut').src = res.image
        setFormula(`z_out = ${w.toFixed(2)} * z_B + ${(1 - w).toFixed(2)} * z_A`)
        highlightThumb(w)
      }
    } catch (e) {
      setInterpMsg('Interpolation error: ' + e.message)
    }
  }, 120)
}

function clearOut() {
  el('imgOut').src = ''
  setFormula('z_out = —')
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Image A controls
  el('genA').addEventListener('click',  () => generate('A'))
  el('backA').addEventListener('click', () => navigate('A', -1))
  el('fwdA').addEventListener('click',  () => navigate('A', +1))

  // Image B controls
  el('genB').addEventListener('click',  () => generate('B'))
  el('backB').addEventListener('click', () => navigate('B', -1))
  el('fwdB').addEventListener('click',  () => navigate('B', +1))

  // Arithmetic
  el('opAdd').addEventListener('click',   () => doOp('add'))
  el('opSubAB').addEventListener('click', () => doOp('subtract_ab'))
  el('opSubBA').addEventListener('click', () => doOp('subtract_ba'))

  // Interpolation
  el('weight').addEventListener('input', ev => {
    setWeightVal(ev.target.value)
    triggerWeightedUpdate()
  })
  el('doInterp').addEventListener('click', doInterp)
  el('clearOut').addEventListener('click', clearOut)

  setWeightVal(el('weight').value)
  setFormula('z_out = —')
  refreshAllButtons()

  // Auto-generate initial pair
  ;(async () => {
    try {
      setMsg('Generating initial images…')
      await generate('A')
      await generate('B')
      await getFilmstrip()
      triggerWeightedUpdate()
      setMsg('')
    } catch (err) {
      setMsg('Initial generation failed: ' + (err.message || err))
    }
  })()
})
