# Report: Eksplorasi Mekanisme Cross-Anatomical Understanding untuk Lightweight General VLM


## 1. Problem utama implementasi cross-anatomical VLM di penelitian saya

Masalah utamanya bukan cuma "apakah model bisa membaca X-ray/CT di berbagai bagian tubuh", tapi **bagaimana satu model ringan (0.5–3B) bisa generalize ke banyak region anatomi tanpa fitur antar-region saling bercampur (entangled)**, sekaligus tetap **feasible dilatih di GPU terbatas (Kaggle T4×2)**.

Kalau model dilatih naif dengan data multi-region yang di-*pool* begitu saja, risiko yang muncul:

- Representasi kepala, dada, abdomen, panggul saling "mencemari" satu sama lain (feature entanglement)
- Performa terlihat oke secara rata-rata, tapi drop di region tertentu yang under-represented
- Model gagal generalize ke region/kondisi yang tidak terlihat saat training (domain shift)

Problem yang ingin dieksplorasi:

> **Mekanisme apa yang bisa menjaga model lightweight tetap bisa membedakan & memahami struktur spesifik tiap region anatomi, tanpa harus menambah beban komputasi yang membuatnya tidak feasible di budget GPU terbatas?**

Jadi fokusnya bukan sekadar "model mana paling akurat lintas region", tapi **trade-off antara mekanisme anti-entanglement vs biaya komputasi tambahan yang mekanisme itu perlukan**.

---

## 2. Exploration yang dilakukan

Tiga paper yang relevan ditemukan, masing-masing mewakili strategi anti-entanglement yang berbeda:

### **A. ViSD-Boost (dekomposisi per-organ sebelum alignment)**
Cao et al., *ICCV 2025* ([arXiv:2508.03742](https://arxiv.org/pdf/2508.03742))

Problem yang diangkat: menyelaraskan gambar CT (SNR rendah) dengan laporan radiologi (SNR tinggi) menimbulkan *semantic density gap* yang bikin alignment bias sinyal dari organ berbeda saling tertimpa saat direpresentasikan secara global.

### **B. RAU (generalisasi via referensi in-context, bukan hafalan bobot)**
Li et al., *arXiv preprint, 2025* ([arXiv:2509.22404](https://arxiv.org/abs/2509.22404))

Problem yang diangkat: VLM umum lemah dalam *reference-based spatial reasoning* dan lokalisasi presisi-tinggi, ditambah data anatomi berlabel ahli yang langka.

### **C. REN (routing eksplisit per-region via Mixture-of-Experts)**
Peltekian et al., *arXiv preprint, 2025 (rev. 2026)* ([arXiv:2510.04923](https://arxiv.org/pdf/2509.22404))

Problem yang diangkat: MoE konvensional mengasumsikan expert homogen & routing agnostik-domain — asumsi yang tidak cocok untuk imaging medis di mana pola patologi sangat bergantung pada struktur anatomi lokal (per lobus paru, dst).

### Dataset yang dipakai di ketiga paper

| Paper | Dataset training | Dataset evaluasi |
|---|---|---|
| ViSD-Boost | CT-RATE, Rad-ChestCT (CT dada), MedVL-CT69K (CT abdomen) | 54 penyakit / 15 organ, zero-shot |
| RAU | RAOS-CT, Arcade-X-Ray (in-distribution) | LERA-X-Ray, CAMUS-Ultrasound (out-of-distribution) |
| REN | Kohort 597 pasien / 1.898 scan CT dada (ILD) | Patient-level cross-validation |

---

## 3. Pendekatan detail masing-masing paper

Ketiganya sama-sama menghindari "satu representasi global untuk semua region", tapi lewat titik intervensi yang berbeda dalam pipeline:

```
Gambar Medis
     ↓
[A] Dekomposisi per-organ SEBELUM alignment   ← ViSD-Boost
     ↓ atau
[B] Referensi in-context SAAT inferensi        ← RAU
     ↓ atau
[C] Routing eksplisit ke expert per-region     ← REN
     ↓
Representasi / Output
```

### A. ViSD-Boost (dekomposisi + normality modeling)
1. Segmentasi tubuh memecah gambar jadi struktur per-organ; LLM memecah laporan jadi sub-laporan per-anatomi (hal. ±3)
2. *Disease-level contrastive learning*: memisahkan sampel normal vs abnormal **per organ**, bukan per gambar secara global (hal. ±4)
3. VQ-VAE mempelajari distribusi organ sehat → gap rekonstruksi dipakai memperkuat sinyal abnormal (hal. ±4)
4. Hasil: AUC rata-rata **84,9%** zero-shot di 54 penyakit/15 organ, mengungguli CLIP, LOVT, MGCA, Imitate, PRIOR, ASG (Tabel hasil, hal. ±6–7)

### B. RAU (one-shot in-context reference)
1. Satu gambar referensi (sudah berlabel) disisipkan ke prompt bersama gambar target (hal. ±3)
2. VLM belajar *relative spatial reasoning* antara reference–target, divalidasi lewat VQA & bounding-box prediction (hal. ±3–4)
3. Output spasial digabung dengan SAM2 untuk segmentasi piksel halus struktur kecil, mis. segmen pembuluh darah (hal. ±4–5)
4. Hasil: konsisten mengungguli baseline *SAM2 fine-tuning* di 2 dataset in-distribution & 2 dataset out-of-distribution (Section Experiments, hal. ±6–7)

### C. REN (Mixture-of-Experts ber-prior anatomi)
1. 7 expert khusus, masing-masing untuk satu lobus paru/kombinasi bilateral (hal. ±2–3)
2. *Gating* multi-modal menggabungkan biomarker radiomik + fitur CNN/ViT/Mamba untuk bobot tiap expert (hal. ±3–4)
3. Hasil: ensemble radiomik-guided AUC **0,8646 ± 0,0467**, +12,5% dari baseline SwinUNETR tunggal (AUC 0,7685, p=0,031); expert lobus bawah capai AUC 0,88–0,90 (Section Results, hal. ±5–6)

---

## 4. Potential adaptation ke topik penelitian 

### A. Lightweight cross-anatomical VLM dengan dekomposisi region ringan
Ketiga paper di atas mahal secara komputasi (butuh segmentasi tubuh tambahan, SAM2, atau 7 expert terpisah), tidak langsung feasible di T4×2. Untuk versi ringan:

> **Bisakah dekomposisi per-region "murah" (misalnya lewat label region di metadata QA, bukan segmentasi model terpisah) tetap mengurangi entanglement pada VLM 0.5–3B?**

### B. Eksplorasi granularitas dekomposisi
Mengadaptasi ide ViSD-Boost tanpa pipeline segmentasi penuh:

```
Region-tagged Loss/Attention
       vs
Global Pooled Loss (baseline naif)
```

Tujuannya mencari: **seberapa halus tagging region (per-organ vs per-kuadran tubuh) yang masih memberi manfaat, tanpa menambah komponen model baru.**

### C. Eksplorasi mekanisme referensi in-context ala RAU sebagai RAG-lite
Karena fine-tuning penuh mahal, referensi in-context RAU bisa diadaptasi jadi skema **retrieval-augmented saat inferensi**: ambil 1 contoh gambar+jawaban per region dari database kecil (≤6.000 gambar), sisipkan ke prompt. 

> Apakah retrieval exemplar per-region saat inferensi bisa menggantikan sebagian kebutuhan fine-tuning lintas-region?

### D. MoE-lite sebagai pembanding konseptual (bukan direplikasi penuh)
REN terlalu berat untuk direplikasi langsung (7 expert penuh), tapi bisa dijadikan **kontrol pembanding**: LoRA-per-kelompok-region (mis. 1 adapter untuk kepala+dada, 1 lagi untuk abdomen+panggul) vs 1 adapter tunggal untuk semua region — mengetes apakah *routing eksplisit ringan* memberi manfaat serupa MoE tanpa biaya penuhnya.

### E. Tambahkan computational evaluation (gap dari ketiga paper)
Ketiganya fokus di clinical performance (AUC/akurasi), **tidak melaporkan** detail:
- peak VRAM saat training/inferensi,
- latency per gambar,
- GPU-seconds/image,
- estimasi biaya untuk 1.000 gambar.

Untuk penelitian saya, evaluasi akan melihat dari:

| Metric | Baseline naif (pooled) | Region-tagged | RAG-lite (referensi) |
| --- | --- | --- | --- |
| Akurasi cross-region | ? | ? | ? |
| Peak VRAM | ? | ? | ? |
| Latency/gambar | ? | ? | ? |
| GPU-sec/image | ? | ? | ? |
| Estimasi biaya/1.000 gambar | ? | ? | ? |
