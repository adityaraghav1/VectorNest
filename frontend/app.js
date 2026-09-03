const API_BASE_URL = "http://127.0.0.1:8000";

const navItems = document.querySelectorAll(".nav-item");
const sections = document.querySelectorAll(".page-section");
const pageTitle = document.getElementById("page-title");

const apiStatusDot = document.getElementById(
    "api-status-dot"
);

const apiStatusText = document.getElementById(
    "api-status-text"
);

const collectionForm = document.getElementById(
    "collection-form"
);

const collectionResult = document.getElementById(
    "collection-result"
);
const ingestionForm = document.getElementById(
    "ingestion-form"
);

const ingestionCollection = document.getElementById(
    "ingestion-collection"
);

const documentIdInput = document.getElementById(
    "document-id"
);

const documentContent = document.getElementById(
    "document-content"
);

const wordCount = document.getElementById(
    "word-count"
);

const metadataContainer = document.getElementById(
    "metadata-container"
);

const addMetadataButton = document.getElementById(
    "add-metadata-button"
);

const ingestionResult = document.getElementById(
    "ingestion-result"
);

const ingestButton = document.getElementById(
    "ingest-button"
);

const semanticSearchForm = document.getElementById(
    "semantic-search-form"
);

const searchCollection = document.getElementById(
    "search-collection"
);

const semanticQuery = document.getElementById(
    "semantic-query"
);

const searchIndex = document.getElementById(
    "search-index"
);

const searchMetric = document.getElementById(
    "search-metric"
);

const searchK = document.getElementById(
    "search-k"
);

const addSearchFilterButton = document.getElementById(
    "add-search-filter-button"
);

const searchFilterContainer = document.getElementById(
    "search-filter-container"
);

const semanticSearchButton = document.getElementById(
    "semantic-search-button"
);

const semanticSearchResults = document.getElementById(
    "semantic-search-results"
);

const vectorCanvas = document.getElementById(
    "vector-space-canvas"
);

const refreshProjectionButton = document.getElementById(
    "refresh-projection-button"
);

const vectorCount = document.getElementById(
    "vector-count"
);

const plotEmptyState = document.getElementById(
    "plot-empty-state"
);

const vectorTooltip = document.getElementById(
    "vector-tooltip"
);

const ragForm = document.getElementById(
    "rag-form"
);

const ragQuestion = document.getElementById(
    "rag-question"
);

const ragCollection = document.getElementById(
    "rag-collection"
);

const ragIndex = document.getElementById(
    "rag-index"
);

const ragMetric = document.getElementById(
    "rag-metric"
);

const ragK = document.getElementById(
    "rag-k"
);

const ragSubmitButton = document.getElementById(
    "rag-submit-button"
);

const ragMessages = document.getElementById(
    "rag-messages"
);

const ragSources = document.getElementById(
    "rag-sources"
);

const ragSourceCount = document.getElementById(
    "rag-source-count"
);

const ragStatus = document.getElementById(
    "rag-status"
);

const vectorContext = vectorCanvas.getContext("2d");

let renderedVectorPoints = [];
let lastSearchResultIds = new Set();    

function switchSection(sectionName) {
    sections.forEach((section) => {
        section.classList.remove("active");
    });

    navItems.forEach((item) => {
        item.classList.remove("active");
    });

    const targetSection = document.getElementById(
        `${sectionName}-section`
    );

    const targetNavItem = document.querySelector(
        `[data-section="${sectionName}"]`
    );

    if (targetSection) {
        targetSection.classList.add("active");
    }

    if (targetNavItem) {
        targetNavItem.classList.add("active");
    }

    if (sectionName === "search") {
    loadProjection();
    }

    const titles = {
        dashboard: "Dashboard",
        collections: "Collections",
        ingestion: "Document Ingestion",
        search: "Semantic Search",
    };

    pageTitle.textContent =
        titles[sectionName] || "VectorNest";
}


navItems.forEach((item) => {
    item.addEventListener("click", () => {
        switchSection(item.dataset.section);
    });
});

function updateWordCount() {
    const text = documentContent.value.trim();

    if (!text) {
        wordCount.textContent = "0 words";
        return;
    }

    const words = text
        .split(/\s+/)
        .filter(Boolean);

    wordCount.textContent =
        `${words.length} word${words.length === 1 ? "" : "s"}`;
}

function addMetadataRow() {
    const row = document.createElement("div");

    row.className = "metadata-row";

    row.innerHTML = `
        <input
            type="text"
            class="metadata-key"
            placeholder="Key e.g. category"
        >

        <input
            type="text"
            class="metadata-value"
            placeholder="Value e.g. programming"
        >

        <button
            type="button"
            class="remove-metadata-button"
            aria-label="Remove metadata"
        >
            ×
        </button>
    `;

    const removeButton = row.querySelector(
        ".remove-metadata-button"
    );

    removeButton.addEventListener(
        "click",
        () => {
            row.remove();
        }
    );

    metadataContainer.appendChild(row);
}


addMetadataButton.addEventListener(
    "click",
    addMetadataRow
);
function collectMetadata() {
    const metadata = {};

    const rows = metadataContainer.querySelectorAll(
        ".metadata-row"
    );

    rows.forEach((row) => {
        const key = row
            .querySelector(".metadata-key")
            .value
            .trim();

        const value = row
            .querySelector(".metadata-value")
            .value
            .trim();

        if (key && value) {
            metadata[key] = value;
        }
    });

    return metadata;
}


documentContent.addEventListener(
    "input",
    updateWordCount
);  

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function renderIngestionSuccess(data) {
    ingestionResult.innerHTML = `
        <div class="ingestion-success">

            <span class="result-badge">
                DOCUMENT INGESTED
            </span>

            <div class="result-stat">
                <span>Document ID</span>
                <strong>
                    ${escapeHtml(data.document_id)}
                </strong>
            </div>

            <div class="result-stat">
                <span>Chunks Created</span>
                <strong>
                    ${data.chunks_created}
                </strong>
            </div>

            <div>
                <p class="eyebrow">
                    STORED RECORDS
                </p>

                <div class="record-list">
                    ${data.record_ids
                        .map(
                            (recordId) => `
                                <div class="record-item">
                                    ${escapeHtml(recordId)}
                                </div>
                            `
                        )
                        .join("")}
                </div>
            </div>
        </div>
    `;
}

ingestionForm.addEventListener(
    "submit",
    async (event) => {
        event.preventDefault();

        const collectionName =
            ingestionCollection.value.trim();

        const documentText =
            documentContent.value.trim();

        const documentId =
            documentIdInput.value.trim();

        if (!collectionName || !documentText) {
            return;
        }

        const payload = {
            document: documentText,
            metadata: collectMetadata(),
        };

        if (documentId) {
            payload.document_id = documentId;
        }

        ingestButton.disabled = true;
        ingestButton.textContent = "Ingesting...";

        ingestionResult.innerHTML = `
            <p class="empty-result">
                Chunking document and generating embeddings...
            </p>
        `;

        try {
            const response = await fetch(
                `${API_BASE_URL}/collections/` +
                `${encodeURIComponent(collectionName)}` +
                `/documents`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json",
                    },

                    body: JSON.stringify(payload),
                }
            );

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    typeof data.detail === "string"
                        ? data.detail
                        : "Document ingestion failed."
                );
            }

            renderIngestionSuccess(data);

        } catch (error) {
            ingestionResult.innerHTML = `
                <p class="error-message">
                    ${escapeHtml(error.message)}
                </p>
            `;

        } finally {
            ingestButton.disabled = false;
            ingestButton.textContent =
                "Ingest Document";
        }
    }
);

function addSearchFilterRow() {
    const row = document.createElement("div");

    row.className = "metadata-row";

    row.innerHTML = `
        <input
            type="text"
            class="search-filter-key"
            placeholder="Key e.g. topic"
        >

        <input
            type="text"
            class="search-filter-value"
            placeholder="Value e.g. python"
        >

        <button
            type="button"
            class="remove-metadata-button"
            aria-label="Remove filter"
        >
            ×
        </button>
    `;

    row.querySelector(
        ".remove-metadata-button"
    ).addEventListener(
        "click",
        () => row.remove()
    );

    searchFilterContainer.appendChild(row);
}


addSearchFilterButton.addEventListener(
    "click",
    addSearchFilterRow
);


function collectSearchFilters() {
    const filters = {};

    const rows = searchFilterContainer.querySelectorAll(
        ".metadata-row"
    );

    rows.forEach((row) => {
        const key = row
            .querySelector(".search-filter-key")
            .value
            .trim();

        const value = row
            .querySelector(".search-filter-value")
            .value
            .trim();

        if (key && value) {
            filters[key] = value;
        }
    });

    return filters;
}

function renderSearchResults(data) {
    if (!data.results || data.results.length === 0) {
        semanticSearchResults.innerHTML = `
            <div class="panel">
                <p class="empty-result">
                    No matching records found.
                </p>
            </div>
        `;

        return;
    }

    lastSearchResultIds = new Set(
    data.results.map(
        (result) => result.id
    )
);

    const resultCards = data.results
        .map((result, index) => {
            const metadataEntries =
                Object.entries(result.metadata || {});

            const metadataHtml = metadataEntries
                .map(
                    ([key, value]) => `
                        <span class="metadata-chip">
                            ${escapeHtml(key)}:
                            ${escapeHtml(value)}
                        </span>
                    `
                )
                .join("");

            return `
                <article class="search-result-card">
                    <div class="search-result-header">
                        <div class="search-result-rank">
                            <span class="rank-badge">
                                ${index + 1}
                            </span>

                            <div>
                                <strong>
                                    Match ${index + 1}
                                </strong>

                                <div class="result-record-id">
                                    ${escapeHtml(result.id)}
                                </div>
                            </div>
                        </div>

                        <div class="search-score">
                            <span>Score</span>
                            <strong>
                                ${Number(result.score).toFixed(6)}
                            </strong>
                        </div>
                    </div>

                    <div class="search-document">
                        ${escapeHtml(
                            result.document || "No document text"
                        )}
                    </div>

                    <div class="search-metadata">
                        ${metadataHtml}
                    </div>
                </article>
            `;
        })
        .join("");

    semanticSearchResults.innerHTML = `
        <div class="panel">
            <div class="search-summary">
                <div>
                    <span class="stat-label">
                        RESULTS
                    </span>

                    <strong>
                        ${data.results.length}
                        matching chunk${data.results.length === 1 ? "" : "s"}
                    </strong>
                </div>

                <div>
                    <span class="stat-label">
                        INDEX
                    </span>

                    <strong>
                        ${escapeHtml(searchIndex.value)}
                    </strong>
                </div>

                <div>
                    <span class="stat-label">
                        METRIC
                    </span>

                    <strong>
                        ${escapeHtml(searchMetric.value)}
                    </strong>
                </div>
            </div>

            <div class="search-result-grid">
                ${resultCards}
            </div>
        </div>
    `;
}

semanticSearchForm.addEventListener(
    "submit",
    async (event) => {
        event.preventDefault();

        const collectionName =
            searchCollection.value.trim();

        const query =
            semanticQuery.value.trim();

        if (!collectionName || !query) {
            return;
        }

        const metadataFilter =
            collectSearchFilters();

        const payload = {
            query,
            metric: searchMetric.value,
            index_type: searchIndex.value,
            k: Number(searchK.value),
        };

        if (
            Object.keys(metadataFilter).length > 0
        ) {
            payload.metadata_filter =
                metadataFilter;
        }

        semanticSearchButton.disabled = true;

        semanticSearchButton.textContent =
            "Searching...";

        semanticSearchResults.innerHTML = `
            <div class="panel">
                <p class="empty-result">
                    Embedding query and searching vectors...
                </p>
            </div>
        `;

        try {
            const response = await fetch(
                `${API_BASE_URL}/collections/` +
                `${encodeURIComponent(collectionName)}` +
                `/semantic-search`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json",
                    },

                    body: JSON.stringify(payload),
                }
            );

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    typeof data.detail === "string"
                        ? data.detail
                        : "Semantic search failed."
                );
            }

            renderSearchResults(data);

            await loadProjection();

        } catch (error) {
            semanticSearchResults.innerHTML = `
                <div class="panel">
                    <p class="error-message">
                        ${escapeHtml(error.message)}
                    </p>
                </div>
            `;

        } finally {
            semanticSearchButton.disabled = false;

            semanticSearchButton.textContent =
                "Run Semantic Search";
        }
    }
);

async function loadProjection() {
    const collectionName =
        searchCollection.value.trim();

    const query =
        semanticQuery.value.trim();

    if (!collectionName) {
        return;
    }

    try {
        let url =
            `${API_BASE_URL}/collections/` +
            `${encodeURIComponent(collectionName)}` +
            `/projection`;

        if (query) {
            url +=
                `?query=${encodeURIComponent(query)}`;
        }

        const response = await fetch(url);

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                typeof data.detail === "string"
                    ? data.detail
                    : "Could not load projection."
            );
        }

        drawVectorSpace(data.points || []);

    } catch (error) {
        clearVectorSpace();

        plotEmptyState.textContent =
            error.message;

        plotEmptyState.classList.remove("hidden");
    }
}

function clearVectorSpace() {
    const rect =
        vectorCanvas.getBoundingClientRect();

    const scale =
        window.devicePixelRatio || 1;

    vectorCanvas.width =
        Math.max(1, Math.floor(rect.width * scale));

    vectorCanvas.height =
        Math.max(1, Math.floor(rect.height * scale));

    vectorContext.setTransform(
        scale,
        0,
        0,
        scale,
        0,
        0
    );

    vectorContext.clearRect(
        0,
        0,
        rect.width,
        rect.height
    );

    renderedVectorPoints = [];

    vectorCount.textContent = "0 vectors";
}

function enforceIndexMetric(
    indexElement,
    metricElement
) {
    if (
        indexElement.value ===
        "kd_tree"
    ) {
        metricElement.value =
            "euclidean";
    }
}


searchIndex.addEventListener(
    "change",
    () => {
        enforceIndexMetric(
            searchIndex,
            searchMetric
        );
    }
);


ragIndex.addEventListener(
    "change",
    () => {
        enforceIndexMetric(
            ragIndex,
            ragMetric
        );
    }
);

searchIndex.addEventListener(
    "change",
    () => {
        enforceIndexMetric(
            searchIndex,
            searchMetric
        );

        ragIndex.value =
            searchIndex.value;

        enforceIndexMetric(
            ragIndex,
            ragMetric
        );
    }
);


ragIndex.addEventListener(
    "change",
    () => {
        enforceIndexMetric(
            ragIndex,
            ragMetric
        );

        searchIndex.value =
            ragIndex.value;

        enforceIndexMetric(
            searchIndex,
            searchMetric
        );
    }
);


searchMetric.addEventListener(
    "change",
    () => {
        if (
            searchIndex.value !==
            "kd_tree"
        ) {
            ragMetric.value =
                searchMetric.value;
        }
    }
);


ragMetric.addEventListener(
    "change",
    () => {
        if (
            ragIndex.value !==
            "kd_tree"
        ) {
            searchMetric.value =
                ragMetric.value;
        }
    }
);

function drawVectorSpace(points) {
    clearVectorSpace();

    if (!points.length) {
        plotEmptyState.textContent =
            "No vectors to visualize yet.";

        plotEmptyState.classList.remove("hidden");

        return;
    }

    plotEmptyState.classList.add("hidden");

    const rect =
        vectorCanvas.getBoundingClientRect();

    const width = rect.width;
    const height = rect.height;

    const padding = 50;

    const xValues = points.map(
        (point) => point.x
    );

    const yValues = points.map(
        (point) => point.y
    );

    let minX = Math.min(...xValues);
    let maxX = Math.max(...xValues);

    let minY = Math.min(...yValues);
    let maxY = Math.max(...yValues);

    if (minX === maxX) {
        minX -= 1;
        maxX += 1;
    }

    if (minY === maxY) {
        minY -= 1;
        maxY += 1;
    }

    const xRange = maxX - minX;
    const yRange = maxY - minY;

    const mapX = (x) =>
        padding +
        ((x - minX) / xRange) *
        (width - padding * 2);

    const mapY = (y) =>
        height -
        padding -
        ((y - minY) / yRange) *
        (height - padding * 2);

    drawAxes(
        width,
        height,
        padding
    );

    renderedVectorPoints = points.map(
        (point) => {
            const canvasX = mapX(point.x);
            const canvasY = mapY(point.y);

            drawVectorPoint(
                canvasX,
                canvasY,
                point
            );

            return {
                ...point,
                canvasX,
                canvasY,
            };
        }
    );

    const documentCount = points.filter(
        (point) => !point.is_query
    ).length;

vectorCount.textContent =
    `${documentCount} stored vector` +
    `${documentCount === 1 ? "" : "s"}`;
}

function drawAxes(
    width,
    height,
    padding
) {
    vectorContext.save();

    vectorContext.strokeStyle =
        "rgba(255, 255, 255, 0.07)";

    vectorContext.lineWidth = 1;

    vectorContext.beginPath();

    vectorContext.moveTo(
        padding,
        height - padding
    );

    vectorContext.lineTo(
        width - padding,
        height - padding
    );

    vectorContext.moveTo(
        padding,
        padding
    );

    vectorContext.lineTo(
        padding,
        height - padding
    );

    vectorContext.stroke();

    vectorContext.restore();
}

function drawVectorPoint(
    x,
    y,
    point
) {
    vectorContext.save();

    const isMatch =
        lastSearchResultIds.has(point.id);

    let fillColor = "#7c8cff";
    let radius = 6;
    let glowColor = "#7c8cff";

    if (isMatch) {
        fillColor = "#f6c85f";
        glowColor = "#f6c85f";
        radius = 7;
    }

    if (point.is_query) {
        fillColor = "#43d6a4";
        glowColor = "#43d6a4";
        radius = 9;
    }

    vectorContext.beginPath();

    vectorContext.arc(
        x,
        y,
        radius,
        0,
        Math.PI * 2
    );

    vectorContext.fillStyle = fillColor;

    vectorContext.shadowBlur = 18;
    vectorContext.shadowColor = glowColor;

    vectorContext.fill();

    vectorContext.strokeStyle =
        "rgba(255,255,255,0.7)";

    vectorContext.lineWidth = 1;

    vectorContext.stroke();

    vectorContext.restore();

    drawVectorLabel(
        x,
        y,
        point,
        isMatch
    );
}

function drawVectorLabel(
    x,
    y,
    point,
    isMatch
) {
    vectorContext.save();

    let labelColor =
        "rgba(220, 226, 240, 0.72)";

    if (isMatch) {
        labelColor = "#f6c85f";
    }

    if (point.is_query) {
        labelColor = "#43d6a4";
    }

    vectorContext.fillStyle =
        labelColor;

    vectorContext.font =
        point.is_query
            ? "600 11px system-ui"
            : "10px system-ui";

    const rawLabel =
        point.is_query
            ? "QUERY"
            : point.id;

    const label =
        rawLabel.length > 22
            ? `${rawLabel.slice(0, 19)}...`
            : rawLabel;

    const direction =
        x > vectorCanvas.getBoundingClientRect().width * 0.72
            ? -1
            : 1;

    const textWidth =
        vectorContext.measureText(label).width;

    const labelX =
        direction === 1
            ? x + 10
            : x - textWidth - 10;

    vectorContext.fillText(
        label,
        labelX,
        y - 10
    );

    vectorContext.restore();
}

vectorCanvas.addEventListener(
    "mousemove",
    (event) => {
        const rect =
            vectorCanvas.getBoundingClientRect();

        const mouseX =
            event.clientX - rect.left;

        const mouseY =
            event.clientY - rect.top;

        const hoveredPoint =
            renderedVectorPoints.find(
                (point) => {
                    const dx =
                        point.canvasX - mouseX;

                    const dy =
                        point.canvasY - mouseY;

                    return Math.sqrt(
                        dx * dx + dy * dy
                    ) <= 10;
                }
            );

        if (!hoveredPoint) {
            vectorTooltip.style.display =
                "none";

            return;
        }

        const preview =
            hoveredPoint.document
                ? hoveredPoint.document.slice(
                    0,
                    160
                )
                : "Query vector";

        vectorTooltip.innerHTML = `
            <strong>
                ${escapeHtml(hoveredPoint.id)}
            </strong>

            <br>

            ${escapeHtml(preview)}
        `;

        vectorTooltip.style.display =
            "block";

        vectorTooltip.style.left =
            `${mouseX + 14}px`;

        vectorTooltip.style.top =
            `${mouseY + 14}px`;
    }
);


vectorCanvas.addEventListener(
    "mouseleave",
    () => {
        vectorTooltip.style.display =
            "none";
    }
);

refreshProjectionButton.addEventListener(
    "click",
    loadProjection
);

function addRagMessage(
    role,
    message
) {
    const messageElement =
        document.createElement("div");

    messageElement.className =
        `rag-message ${
            role === "user"
                ? "user-message"
                : "assistant-message"
        }`;

    const avatar =
        role === "user"
            ? "YOU"
            : "VN";

    messageElement.innerHTML = `
        <div class="rag-avatar">
            ${avatar}
        </div>

        <div class="rag-message-content">
            ${escapeHtml(message)}
        </div>
    `;

    ragMessages.appendChild(
        messageElement
    );

    ragMessages.scrollTop =
        ragMessages.scrollHeight;

    return messageElement;
}


function addRagLoadingMessage() {
    const messageElement =
        document.createElement("div");

    messageElement.className =
        "rag-message assistant-message";

    messageElement.innerHTML = `
        <div class="rag-avatar">
            VN
        </div>

        <div class="rag-message-content">
            <div class="rag-loading-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;

    ragMessages.appendChild(
        messageElement
    );

    ragMessages.scrollTop =
        ragMessages.scrollHeight;

    return messageElement;
}


function updateRagStatus(
    message,
    state = "ready"
) {
    ragStatus.textContent = message;

    ragStatus.classList.remove(
        "loading",
        "error"
    );

    if (state === "loading") {
        ragStatus.classList.add(
            "loading"
        );
    }

    if (state === "error") {
        ragStatus.classList.add(
            "error"
        );
    }
}


function renderRagSources(
    sources
) {
    if (!sources || sources.length === 0) {
        ragSourceCount.textContent =
            "0 sources";

        ragSources.innerHTML = `
            <p class="empty-result">
                No supporting chunks were retrieved.
            </p>
        `;

        return;
    }

    ragSourceCount.textContent =
        `${sources.length} source${
            sources.length === 1
                ? ""
                : "s"
        }`;

    ragSources.innerHTML = sources
        .map((source) => {
            const metadataEntries =
                Object.entries(
                    source.metadata || {}
                );

            const metadataHtml =
                metadataEntries
                    .map(
                        ([key, value]) => `
                            <span>
                                ${escapeHtml(key)}:
                                ${escapeHtml(value)}
                            </span>
                        `
                    )
                    .join("");

            return `
                <article class="rag-source-card">
                    <div class="rag-source-header">
                        <div class="rag-source-id">
                            ${escapeHtml(source.id)}
                        </div>

                        <div class="rag-source-score">
                            ${Number(
                                source.score
                            ).toFixed(6)}
                        </div>
                    </div>

                    <div class="rag-source-document">
                        ${escapeHtml(
                            source.document ||
                            "No document text"
                        )}
                    </div>

                    ${
                        metadataHtml
                            ? `
                                <div
                                    class="rag-source-metadata"
                                >
                                    ${metadataHtml}
                                </div>
                            `
                            : ""
                    }
                </article>
            `;
        })
        .join("");
}

ragQuestion.addEventListener(
    "keydown",
    (event) => {
        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {
            event.preventDefault();

            if (
                !ragSubmitButton.disabled
            ) {
                ragForm.requestSubmit();
            }
        }
    }
);


ragForm.addEventListener(
    "submit",
    async (event) => {
        event.preventDefault();

        const collectionName =
            ragCollection.value.trim();

        const question =
            ragQuestion.value.trim();

        if (
            !collectionName ||
            !question
        ) {
            return;
        }

        const payload = {
            question,
            metric: ragMetric.value,
            index_type: ragIndex.value,
            k: Number(ragK.value),
        };

addRagMessage(
    "user",
    question
);

ragQuestion.value = "";

/*
 * Clear sources from the previous question
 * while the new retrieval is running.
 */
ragSourceCount.textContent =
    "Retrieving...";

ragSources.innerHTML = `
    <p class="empty-result">
        Searching for relevant chunks...
    </p>
`;

ragSubmitButton.disabled = true;

        ragSubmitButton.textContent =
            "Thinking...";

        updateRagStatus(
            "Retrieving context...",
            "loading"
        );

        const loadingMessage =
            addRagLoadingMessage();

        try {
            const response = await fetch(
                `${API_BASE_URL}/collections/` +
                `${encodeURIComponent(
                    collectionName
                )}/rag/stream`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",
                    },

                    body: JSON.stringify(
                        payload
                    ),
                }
            );

            if (!response.ok) {
                const errorText =
                    await response.text();

                throw new Error(
                    errorText ||
                    "RAG request failed."
                );
            }

            if (!response.body) {
                throw new Error(
                    "Streaming response is unavailable."
                );
            }

            const reader =
                response.body.getReader();

            const decoder =
                new TextDecoder();

            let buffer = "";

            let assistantMessage = null;

            let assistantContent = null;

            let retrievedSources = [];

            let streamFinished = false;

            while (!streamFinished) {
                const {
                    value,
                    done,
                } = await reader.read();

                if (done) {
                    break;
                }

                buffer += decoder.decode(
                    value,
                    {
                        stream: true,
                    }
                );

                const lines =
                    buffer.split("\n");

                buffer =
                    lines.pop() || "";

                for (
                    const line of lines
                ) {
                    const trimmedLine =
                        line.trim();

                    if (!trimmedLine) {
                        continue;
                    }

                    let eventData;

                    try {
                        eventData =
                            JSON.parse(
                                trimmedLine
                            );
                    } catch {
                        continue;
                    }

                    if (
                        eventData.type ===
                        "sources"
                    ) {
                        retrievedSources =
                            eventData.sources ||
                            [];

                        renderRagSources(
                            retrievedSources
                        );

                        updateRagStatus(
                            "Generating answer...",
                            "loading"
                        );

                        continue;
                    }

                    if (
                        eventData.type ===
                        "token"
                    ) {
                        if (!assistantMessage) {
                            loadingMessage.remove();

                            assistantMessage =
                                addRagMessage(
                                    "assistant",
                                    ""
                                );

                            assistantContent =
                                assistantMessage
                                    .querySelector(
                                        ".rag-message-content"
                                    );

                            assistantContent.classList.add(
                                "rag-streaming"
                            );
                        }

                        assistantContent.textContent +=
                            eventData.content ||
                            "";

                        ragMessages.scrollTop =
                            ragMessages.scrollHeight;

                        continue;
                    }

                    if (
                        eventData.type ===
                        "done"
                    ) {
                        streamFinished = true;

                        if (assistantContent) {
                            assistantContent.classList.remove(
                                "rag-streaming"
                            );
                        }
                    }
                }
            }

            if (loadingMessage.isConnected) {
                loadingMessage.remove();
            }

            if (!assistantMessage) {
                addRagMessage(
                    "assistant",
                    (
                        "I could not generate " +
                        "an answer."
                    )
                );
            }

            renderRagSources(
                retrievedSources
            );

            updateRagStatus(
                "Answer grounded in retrieved context"
            );

        } catch (error) {
            if (
                loadingMessage.isConnected
            ) {
                loadingMessage.remove();
            }

            addRagMessage(
                "assistant",
                `Error: ${error.message}`
            );

            renderRagSources([]);

            updateRagStatus(
                "RAG request failed",
                "error"
            );

        } finally {
            ragSubmitButton.disabled =
                false;

            ragSubmitButton.textContent =
                "Ask";
        }
    }
);

function syncCollectionName(
    sourceInput
) {
    const collectionName =
        sourceInput.value.trim();

    if (!collectionName) {
        return;
    }

    if (
        sourceInput !== ingestionCollection
    ) {
        ingestionCollection.value =
            collectionName;
    }

    if (
        sourceInput !== searchCollection
    ) {
        searchCollection.value =
            collectionName;
    }

    if (
        sourceInput !== ragCollection
    ) {
        ragCollection.value =
            collectionName;
    }
}


ingestionCollection.addEventListener(
    "input",
    () => {
        syncCollectionName(
            ingestionCollection
        );
    }
);


searchCollection.addEventListener(
    "input",
    () => {
        syncCollectionName(
            searchCollection
        );
    }
);


ragCollection.addEventListener(
    "input",
    () => {
        syncCollectionName(
            ragCollection
        );
    }
);

async function checkApiHealth() {
    try {
        const response = await fetch(
            `${API_BASE_URL}/health`
        );

        if (!response.ok) {
            throw new Error("API unavailable");
        }

        apiStatusDot.style.background = "#43d6a4";
        apiStatusText.textContent = "API Connected";
    } catch {
        apiStatusDot.style.background = "#ff6b7a";
        apiStatusText.textContent = "API Offline";
    }
}


collectionForm.addEventListener(
    "submit",
    async (event) => {
        event.preventDefault();

        collectionResult.textContent =
            "Creating collection...";

        const payload = {
            name: document
                .getElementById("collection-name")
                .value
                .trim(),

            dimension: Number(
                document.getElementById(
                    "collection-dimension"
                ).value
            ),

            distance_metric: document.getElementById(
                "collection-metric"
            ).value,
        };

        try {
            const response = await fetch(
                `${API_BASE_URL}/collections`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(payload),
                }
            );

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.detail || "Request failed"
                );
            }

            collectionResult.textContent =
                `Collection created successfully.\n\n` +
                JSON.stringify(data, null, 2);
        } catch (error) {
            collectionResult.textContent =
                `Error: ${error.message}`;
        }
    }
);


checkApiHealth();