document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const statusPill = document.getElementById('statusPill');
    const statusText = document.getElementById('statusText');
    const activeDocName = document.getElementById('activeDocName');
    const indexedChunks = document.getElementById('indexedChunks');
    const embeddingModel = document.getElementById('embeddingModel');

    const queryInput = document.getElementById('queryInput');
    const submitQueryBtn = document.getElementById('submitQueryBtn');
    const docUpload = document.getElementById('docUpload');
    const runBenchmarkBtn = document.getElementById('runBenchmarkBtn');

    const loader = document.getElementById('loader');
    const comparisonWorkspace = document.getElementById('comparisonWorkspace');

    // Baseline UI Elements
    const bTokens = document.getElementById('bTokens');
    const bChunks = document.getElementById('bChunks');
    const bLatency = document.getElementById('bLatency');
    const bAnswer = document.getElementById('bAnswer');
    const bChunksList = document.getElementById('bChunksList');

    // QATM UI Elements
    const qTier = document.getElementById('qTier');
    const qSavingsBadge = document.getElementById('qSavingsBadge');
    const qTokens = document.getElementById('qTokens');
    const qSavings = document.getElementById('qSavings');
    const qLatency = document.getElementById('qLatency');
    const qAnswer = document.getElementById('qAnswer');
    const qChunksList = document.getElementById('qChunksList');
    const gradientChart = document.getElementById('gradientChart');

    // Benchmark UI Elements
    const benchmarkSection = document.getElementById('benchmarkSection');
    const closeBenchmarkBtn = document.getElementById('closeBenchmarkBtn');
    const benchmarkKpis = document.getElementById('benchmarkKpis');
    const benchmarkTableBody = document.getElementById('benchmarkTableBody');

    // Preset Pills
    document.querySelectorAll('.pill-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            queryInput.value = btn.getAttribute('data-query');
            executeQuery();
        });
    });

    // Event Listeners
    submitQueryBtn.addEventListener('click', executeQuery);
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') executeQuery();
    });

    docUpload.addEventListener('change', handleDocumentUpload);
    runBenchmarkBtn.addEventListener('click', executeBenchmark);
    closeBenchmarkBtn.addEventListener('click', () => {
        benchmarkSection.classList.add('hidden');
    });

    // Check Backend Health on Boot
    checkHealth();

    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            if (res.ok) {
                const data = await res.json();
                statusText.textContent = "API Engine Ready";
                statusPill.style.borderColor = "rgba(16, 185, 129, 0.4)";
                activeDocName.textContent = data.active_document;
                indexedChunks.textContent = data.indexed_chunks;
                embeddingModel.textContent = data.embedding_model;
            } else {
                statusText.textContent = "Engine Offline";
            }
        } catch (err) {
            statusText.textContent = "API Offline";
            console.error("Health check error:", err);
        }
    }

    async function executeQuery() {
        const qText = queryInput.value.trim();
        if (!qText) return;

        loader.classList.remove('hidden');
        comparisonWorkspace.style.opacity = '0.4';

        try {
            const res = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: qText })
            });

            if (!res.ok) {
                const errData = await res.json();
                alert(`Error executing query: ${errData.detail || 'Server error'}`);
                return;
            }

            const data = await res.json();
            renderResults(data);

        } catch (err) {
            alert(`Failed to connect to API server: ${err.message}`);
            console.error(err);
        } finally {
            loader.classList.add('hidden');
            comparisonWorkspace.style.opacity = '1';
        }
    }

    function renderResults(data) {
        const b = data.baseline;
        const q = data.qatm;

        // Render Baseline Card
        bTokens.textContent = `${b.context_tokens} Tokens`;
        bChunks.textContent = b.retrieved_chunks_count;
        bLatency.textContent = `${b.latency_ms} ms`;
        bAnswer.textContent = b.answer;

        bChunksList.innerHTML = b.retrieved_chunks.map(c => `
            <div class="chunk-item">
                <div class="chunk-meta">
                    <span>${c.id} [${c.granularity}]</span>
                    <span>Similarity: ${c.score}</span>
                </div>
                <div>${escapeHtml(c.text)}</div>
            </div>
        `).join('');

        // Render QATM Card
        qTier.textContent = `Tier ${q.query_tier_code}: ${q.query_tier}`;
        qSavingsBadge.textContent = `-${q.token_savings_percent}% Tokens`;
        qTokens.textContent = q.context_tokens;
        qSavings.textContent = `-${q.token_savings_percent}%`;
        qLatency.textContent = `${q.latency_ms} ms`;
        qAnswer.textContent = q.answer;

        qChunksList.innerHTML = q.retrieved_chunks.map(c => `
            <div class="chunk-item">
                <div class="chunk-meta">
                    <span>${c.id} [${c.granularity}]</span>
                    <span>Similarity: ${c.score}</span>
                </div>
                <div>${escapeHtml(c.text)}</div>
            </div>
        `).join('');

        // Render Score Gradient Bar Chart
        if (q.similarity_dropoffs && q.similarity_dropoffs.length > 0) {
            gradientChart.innerHTML = q.similarity_dropoffs.map(s => {
                const widthPct = Math.min(Math.max(s.delta * 200, 10), 100);
                return `
                    <div class="gradient-step">
                        <span class="step-label">Rank ${s.rank_pair}</span>
                        <div class="step-bar-outer">
                            <div class="step-bar-inner" style="width: ${widthPct}%;"></div>
                        </div>
                        <span class="step-val">Δ ${s.delta}</span>
                    </div>
                `;
            }).join('');
        } else {
            gradientChart.innerHTML = '<div class="empty-chart">Optimal cutoff reached at rank 1</div>';
        }
    }

    async function handleDocumentUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        loader.classList.remove('hidden');
        statusText.textContent = "Indexing Document...";

        try {
            const res = await fetch('/api/ingest', {
                method: 'POST',
                body: formData
            });

            if (res.ok) {
                const data = await res.json();
                activeDocName.textContent = data.document_name;
                indexedChunks.textContent = data.total_chunks;
                alert(`Document '${data.document_name}' indexed successfully (${data.total_chunks} chunks).`);
            } else {
                const err = await res.json();
                alert(`Upload failed: ${err.detail}`);
            }
        } catch (err) {
            alert(`Document upload error: ${err.message}`);
        } finally {
            loader.classList.add('hidden');
            checkHealth();
        }
    }

    async function executeBenchmark() {
        loader.classList.remove('hidden');
        benchmarkSection.classList.add('hidden');

        try {
            const res = await fetch('/api/benchmark', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_path: "data/evaluation/eval_dataset.json" })
            });

            if (res.ok) {
                const report = await res.json();
                renderBenchmarkReport(report);
            } else {
                alert("Failed to run benchmark suite.");
            }
        } catch (err) {
            alert(`Benchmark execution error: ${err.message}`);
        } finally {
            loader.classList.add('hidden');
        }
    }

    function renderBenchmarkReport(report) {
        const s = report.summary;
        const b = s.baseline_stats;
        const q = s.qatm_stats;

        benchmarkKpis.innerHTML = `
            <div class="kpi-card">
                <span class="kpi-val">${s.total_queries_evaluated}</span>
                <span class="kpi-title">Queries Evaluated</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-val">${q.token_savings_percent.mean}%</span>
                <span class="kpi-title">Mean Token Reduction</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-val">${q.context_tokens.mean}</span>
                <span class="kpi-title">Mean QATM Context Tokens</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-val">${q.latency_ms.mean} ms</span>
                <span class="kpi-title">Mean QATM Latency</span>
            </div>
        `;

        benchmarkTableBody.innerHTML = `
            <tr>
                <td><strong>Mean Context Tokens</strong></td>
                <td>${b.context_tokens.mean}</td>
                <td>${q.context_tokens.mean}</td>
                <td><strong style="color: var(--accent-emerald);">-${q.token_savings_percent.mean}%</strong></td>
            </tr>
            <tr>
                <td><strong>P95 Context Tokens</strong></td>
                <td>${b.context_tokens.p95}</td>
                <td>${q.context_tokens.p95}</td>
                <td><strong style="color: var(--accent-emerald);">Pruned Tail Noise</strong></td>
            </tr>
            <tr>
                <td><strong>Mean Total Latency (ms)</strong></td>
                <td>${b.latency_ms.mean} ms</td>
                <td>${q.latency_ms.mean} ms</td>
                <td><strong style="color: var(--primary-cyan);">${Math.round(b.latency_ms.mean - q.latency_ms.mean)} ms diff</strong></td>
            </tr>
            <tr>
                <td><strong>Average Retained Chunks</strong></td>
                <td>${b.avg_retained_chunks} chunks</td>
                <td>${q.avg_retained_chunks} chunks</td>
                <td><strong style="color: var(--accent-purple);">Dynamic $k$ adapt</strong></td>
            </tr>
        `;

        benchmarkSection.classList.remove('hidden');
        benchmarkSection.scrollIntoView({ behavior: 'smooth' });
    }

    function escapeHtml(text) {
        if (!text) return '';
        return text.replace(/&/g, "&amp;")
                   .replace(/</g, "&lt;")
                   .replace(/>/g, "&gt;")
                   .replace(/"/g, "&quot;")
                   .replace(/'/g, "&#039;");
    }
});
