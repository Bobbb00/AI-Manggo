    const menuBtn = document.getElementById('menu-btn');
        const mobileMenu = document.getElementById('mobile-menu');
        const mobileBackdrop = document.getElementById('mobileBackdrop'); // Elemen Backdrop
        const mobileCloseBtn = document.getElementById('mobile-close-btn'); // Tombol Tutup
        const fileInput = document.getElementById('fileInput');
        const uploadBtn = document.getElementById('uploadBtn');
        const previewContainer = document.getElementById('previewContainer');
        const previewImage = document.getElementById('previewImage');
        const resultContainer = document.getElementById('resultContainer');
        const resultLabel = document.getElementById('resultLabel');
        const confidence = document.getElementById('confidence');
        const historyList = document.getElementById('historyList');
        const resetBtn = document.getElementById('resetBtn');
        const openCameraBtn = document.getElementById('openCameraBtn');
        const cameraContainer = document.getElementById('cameraContainer');
        const cameraStreamEl = document.getElementById('cameraStream');
        const captureBtn = document.getElementById('captureBtn');
        const stopCameraBtn = document.getElementById('stopCameraBtn');
        const captureCanvas = document.getElementById('captureCanvas');

        let cameraStream = null;
        let historyData = []; // Data riwayat kosong

        // --- Fungsi Mobile Menu ---
        function openMobileMenu() {
          mobileMenu.classList.remove('translate-x-full');
          mobileMenu.setAttribute('aria-hidden', 'false');
          mobileBackdrop.classList.remove('hidden');
        }

        function closeMobileMenu() {
          mobileMenu.classList.add('translate-x-full');
          mobileMenu.setAttribute('aria-hidden', 'true');
          mobileBackdrop.classList.add('hidden');
        }

        menuBtn.addEventListener('click', () => {
          // Menggunakan logika if-else untuk toggle
          if (mobileMenu.classList.contains('translate-x-full')) openMobileMenu(); else closeMobileMenu();
        });

        mobileBackdrop.addEventListener('click', closeMobileMenu);
        if (mobileCloseBtn) mobileCloseBtn.addEventListener('click', closeMobileMenu);

        // --- Fungsi Upload Gambar ---
        uploadBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', e => {
          const file = e.target.files[0];
          if (file) {
            previewImage.src = URL.createObjectURL(file);
            previewContainer.classList.remove('hidden');
            stopCamera(); // Pastikan kamera mati jika upload file
          }
        });

        // --- Fungsi Kamera ---
        async function startCamera() {
          try {
            cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
            cameraStreamEl.srcObject = cameraStream;
            cameraContainer.classList.remove('hidden');
            document.getElementById('dropZone').classList.add('opacity-60');
          } catch (err) {
            alert('Tidak dapat mengakses kamera. Pastikan izin diberikan dan perangkat memiliki kamera.');
            console.error('startCamera error', err);
          }
        }

        function stopCamera() {
          if (cameraStream) {
            cameraStream.getTracks().forEach(t => t.stop());
            cameraStream = null;
          }
          cameraStreamEl.srcObject = null;
          cameraContainer.classList.add('hidden');
          document.getElementById('dropZone').classList.remove('opacity-60');
        }

        async function capturePhoto() {
          if (!cameraStream) return;
          const video = cameraStreamEl;
          const w = video.videoWidth || 640;
          const h = video.videoHeight || 480;
          captureCanvas.width = w;
          captureCanvas.height = h;
          const ctx = captureCanvas.getContext('2d');
          ctx.drawImage(video, 0, 0, w, h);

          captureCanvas.toBlob(async (blob) => {
            if (!blob) return;
            const file = new File([blob], `capture_${Date.now()}.jpg`, { type: blob.type || 'image/jpeg' });

            const dt = new DataTransfer();
            dt.items.add(file);
            fileInput.files = dt.files;

            previewImage.src = URL.createObjectURL(file);
            previewContainer.classList.remove('hidden');

            stopCamera();
          }, 'image/jpeg', 0.95);
        }

        openCameraBtn.addEventListener('click', startCamera);
        stopCameraBtn.addEventListener('click', stopCamera);
        captureBtn.addEventListener('click', capturePhoto);

        // --- Fungsi Utama: KLASIFIKASI ---
        document.getElementById('classifyBtn').addEventListener('click', async () => {
            if (fileInput.files.length === 0) {
                alert('Mohon pilih atau ambil foto mangga terlebih dahulu.');
                return;
            }

            const file = fileInput.files[0];

            // START LOADING STATE
            const originalText = classifyBtn.textContent;
            classifyBtn.disabled = true;
            classifyBtn.textContent = 'Memproses... ⏳';
            resultContainer.classList.add('hidden'); 

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/classify', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.status !== 200 || result.error) {
                    alert(`Klasifikasi Gagal: ${result.error || 'Terjadi kesalahan server.'}`);
                    return;
                }
                // ❗ VALIDASI: CEK APAKAH INI BUKAN BUAH MANGGA
                if (result.label === "Not Mango" || result.label === "Unknown" || result.is_mango === false) {
                  alert("Ini bukan buah mangga! Mohon unggah gambar buah mangga.");
                  return; // jangan tampilkan hasil & jangan simpan riwayat
                  }

                // Proses dan tampilkan hasil dari server
                const label = result.label;
                const acc = (result.confidence * 100).toFixed(2);
                const predictions = result.predictions;

                resultLabel.textContent = label;
                confidence.textContent = `Akurasi: ${acc}%`;
                resultContainer.classList.remove('hidden');

                // Simpan ke Riwayat Sementara
                const historyItem = {
                    id: `h_${Date.now()}`,
                    waktu: new Date().toLocaleString(),
                    label: label,
                    akurasi: `${acc}%`,
                    predictions: predictions, 
                    imageSrc: previewImage.src || null
                };

                historyData.unshift(historyItem);
                updateHistory();

            } catch (error) {
                console.error('Error saat mengirim data klasifikasi:', error);
                alert('Gagal menghubungi server klasifikasi.');
            } finally {
                // SELALU KEMBALIKAN TOMBOL KE STATUS NORMAL
                classifyBtn.disabled = false;
                classifyBtn.textContent = originalText;
            }
        });


        // --- Fungsi Update Riwayat ---
        function updateHistory() {
          historyList.innerHTML = "";
          if (!historyData || historyData.length === 0) {
            historyList.innerHTML = '<p class="text-center text-gray-400 italic">Belum ada hasil klasifikasi.</p>';
            return;
          }

          historyData.forEach((item, index) => {
            const el = document.createElement("div");
            el.className = "border border-gray-200 rounded-lg p-2 md:p-3 mb-3";

            const header = document.createElement('div');
            header.className = 'flex flex-col xs:flex-row md:flex-row items-start md:items-center justify-between gap-2 md:gap-3';

            const left = document.createElement('div');
            left.className = 'flex items-center gap-2 md:gap-3';

            if (item.imageSrc) {
              const thumb = document.createElement('img');
              thumb.src = item.imageSrc;
              thumb.className = 'w-10 h-10 md:w-12 md:h-12 object-cover rounded-md border';
              thumb.alt = 'thumb';
              left.appendChild(thumb);
            }

            const meta = document.createElement('div');
            meta.className = 'flex flex-col';
            meta.innerHTML = `<span class="font-medium text-sm md:text-base">${index + 1}. ${item.label}</span><span class="text-xs text-gray-500">${item.waktu}</span>`;
            left.appendChild(meta);

            const actions = document.createElement('div');
            actions.className = 'flex flex-col md:flex-row items-stretch md:items-center gap-1 md:gap-2 mt-2 md:mt-0';

            const acc = document.createElement('div');
            acc.className = 'text-xs md:text-sm text-green-700 font-semibold md:mr-2 mb-1 md:mb-0';
            acc.textContent = item.akurasi;

            const detailBtn = document.createElement('button');
            detailBtn.className = 'text-xs md:text-sm bg-blue-600 text-white px-2 py-1 md:px-3 md:py-1 rounded hover:bg-blue-700 mb-1 md:mb-0';
            detailBtn.textContent = 'Lihat Detail';

            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'text-xs md:text-sm bg-red-500 text-white px-2 py-1 md:px-3 md:py-1 rounded hover:bg-red-600';
            deleteBtn.textContent = 'Hapus';

            actions.appendChild(acc);
            actions.appendChild(detailBtn);
            actions.appendChild(deleteBtn);

            header.appendChild(left);
            header.appendChild(actions);

            el.appendChild(header);

            // Detail panel (hidden by default)
            const details = document.createElement('div');
            details.className = 'mt-2 md:mt-3 hidden';
            details.innerHTML = `
              <div class="flex flex-col md:flex-row items-start md:items-center gap-2 md:gap-4">
                <div class="flex-shrink-0 mb-2 md:mb-0">
                  ${item.imageSrc ? `<img src="${item.imageSrc}" class="w-28 h-28 md:w-40 md:h-40 object-cover rounded-md border" alt="img">` : '<div class="w-28 h-28 md:w-40 md:h-40 bg-gray-100 flex items-center justify-center text-xs md:text-sm text-gray-500 rounded-md">No Image</div>'}
                </div>
                <div class="flex-1">
                  <p class="mb-1 md:mb-2 text-sm md:text-base"><strong>Label:</strong> ${item.label}</p>
                  <p class="mb-1 md:mb-2 text-sm md:text-base"><strong>Akurasi:</strong> ${item.akurasi}</p>
                  <p class="mb-1 md:mb-2 text-sm md:text-base"><strong>Waktu:</strong> ${item.waktu}</p>
                  <div>
                    <strong>Prediksi (top):</strong>
                    <ul class="list-disc list-inside text-xs md:text-sm text-gray-700 mt-1">
                      ${item.predictions ? item.predictions.map(p => `<li>${p.label}: ${(p.p*100).toFixed(2)}%</li>`).join('') : '<li>-</li>'}
                    </ul>
                  </div>
                </div>
              </div>
            `;

            el.appendChild(details);

            // Wire buttons
            detailBtn.addEventListener('click', () => {
              details.classList.toggle('hidden');
              detailBtn.textContent = details.classList.contains('hidden') ? 'Lihat Detail' : 'Tutup Detail';
            });

            deleteBtn.addEventListener('click', () => {
              historyData = historyData.filter(h => h.id !== item.id);
              updateHistory();
            });

            historyList.appendChild(el);
          });
        }

        updateHistory();

        resetBtn.addEventListener('click', () => {
          window.location.reload();
        })

        